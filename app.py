"""
Streamlit app: estimate-value sanity check against clustered historical items.

Flow
----
1. User enters an item_label, an estimate_value, and a similarity threshold.
2. We embed item_label with OpenAI text-embedding-3-large (3072 dim).
3. We query Supabase (pgvector) for the top-5 most similar item_labels.
4. We take the top-1 to identify the (sub_cluster, sub_name).
5. We pull cluster_stats and check whether estimate_value falls within
   [lower_bound, upper_bound]. If similarity < threshold OR need_info is TRUE
   we warn the user that the verdict is not reliable.

Setup (run once in Supabase SQL editor):
----------------------------------------
create extension if not exists pg_trgm;

create index if not exists item_embeddings_item_label_trgm_idx
on item_embeddings using gin (item_label gin_trgm_ops);

-- Optional: speed up large_category filter
create index if not exists item_embeddings_large_category_idx
on item_embeddings (large_category);

create index if not exists cluster_stats_large_category_idx
on cluster_stats (large_category);

-- Drop old signature if present (no large_category arg)
drop function if exists match_items_hybrid(halfvec, text, float, float, integer);
drop function if exists match_items_hybrid(halfvec, text, double precision, double precision, integer);

-- Hybrid search filtered by large_category
create or replace function match_items_hybrid(
    query_embedding halfvec(3072),
    query_text text,
    p_large_category text,
    vector_weight float default 0.5,
    keyword_weight float default 0.5,
    match_count int default 5
)
returns table (
    id bigint,
    sub_cluster text,
    sub_name text,
    item_label text,
    similarity float,
    vector_sim float,
    keyword_sim float
)
language sql stable as $$
    with vec_pool as (
        select id from item_embeddings
        where large_category = p_large_category
        order by embedding <=> query_embedding
        limit 50
    ),
    kw_pool as (
        select id from item_embeddings
        where large_category = p_large_category
          and item_label % query_text
        order by similarity(item_label, query_text) desc
        limit 50
    ),
    candidates as (
        select id from vec_pool
        union
        select id from kw_pool
    )
    select
        e.id, e.sub_cluster, e.sub_name, e.item_label,
        (vector_weight * (1 - (e.embedding <=> query_embedding))
         + keyword_weight * similarity(e.item_label, query_text))::float as similarity,
        (1 - (e.embedding <=> query_embedding))::float as vector_sim,
        similarity(e.item_label, query_text)::float    as keyword_sim
    from item_embeddings e
    join candidates c on c.id = e.id
    order by similarity desc
    limit match_count;
$$;

notify pgrst, 'reload schema';
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import gaussian_kde
from supabase import create_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"  # 3072 dim
TOP_K = 5

@st.cache_data(ttl=300)
def fetch_cluster_prices(
    sub_cluster: str, sub_name: str, large_category: str
) -> list[float]:
    """Pull raw price_per_unit from item_embeddings for one cluster."""
    sb = get_supabase_client()
    # Paginate in case a cluster has > 1000 rows (default PostgREST limit)
    prices: list[float] = []
    page_size = 1000
    offset = 0
    while True:
        res = (
            sb.table("item_embeddings")
            .select("price_per_unit")
            .eq("sub_cluster", sub_cluster)
            .eq("sub_name", sub_name)
            .eq("large_category", large_category)
            .not_.is_("price_per_unit", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            break
        for r in rows:
            try:
                prices.append(float(r["price_per_unit"]))
            except (TypeError, ValueError):
                pass
        if len(rows) < page_size:
            break
        offset += page_size
    return prices


@st.cache_data(ttl=300)
def fetch_cluster_items(
    sub_cluster: str, sub_name: str, large_category: str
) -> list[dict]:
    """Pull all items (label + price + flag) of one cluster, paginated."""
    sb = get_supabase_client()
    rows = []
    page_size = 1000
    last_id = 0
    while True:
        res = (
            sb.table("item_embeddings")
            .select("id, item_label, price_per_unit, is_in_hensachi_range, trouble_id")
            .eq("sub_cluster", sub_cluster)
            .eq("sub_name", sub_name)
            .eq("large_category", large_category)
            .gt("id", last_id)
            .order("id", desc=False)
            .limit(page_size)
            .execute()
        )
        batch = res.data or []
        if not batch:
            break
        rows.extend(batch)
        last_id = batch[-1]["id"]
        if len(batch) < page_size:
            break
    return rows


@st.cache_resource
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing in .env")
        st.stop()
    return OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("SUPABASE_URL / SUPABASE_KEY is missing in .env")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def embed(text: str) -> list[float]:
    resp = get_openai_client().embeddings.create(
        model=EMBEDDING_MODEL, input=[text]
    )
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one OpenAI call; preserves input order."""
    resp = get_openai_client().embeddings.create(
        model=EMBEDDING_MODEL, input=texts
    )
    out: list[list[float]] = [None] * len(texts)  # type: ignore
    for item in resp.data:
        out[item.index] = item.embedding
    return out


def search_similar(
    query_embedding: list[float],
    query_text: str,
    large_category: str,
    k: int = TOP_K,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> list[dict[str, Any]]:
    sb = get_supabase_client()
    res = sb.rpc(
        "match_items_hybrid",
        {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "p_large_category": large_category,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "match_count": k,
        },
    ).execute()
    return res.data or []


def get_cluster_stats(
    sub_cluster: str, sub_name: str, large_category: str
) -> dict[str, Any] | None:
    sb = get_supabase_client()
    q = (
        sb.table("cluster_stats")
        .select("*")
        .eq("sub_cluster", sub_cluster)
        .eq("sub_name", sub_name)
        .eq("large_category", large_category)
        .limit(1)
        .execute()
    )
    rows = q.data or []
    return rows[0] if rows else None


def fmt_money(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return str(v)


def compute_verdict(
    item_label: str,
    estimate_value: float,
    large_category: str,
    threshold: float,
    vector_weight: float,
    query_emb: list[float],
) -> dict[str, Any]:
    """Run hybrid search + cluster lookup and return a verdict dict."""
    matches = search_similar(
        query_emb,
        query_text=item_label,
        large_category=large_category,
        k=TOP_K,
        vector_weight=vector_weight,
        keyword_weight=1.0 - vector_weight,
    )
    out: dict[str, Any] = {
        "verdict": "NO REFERENCE",
        "note": None,
        "matches": matches,
        "top_sim": None,
        "stats": None,
    }
    if not matches:
        out["note"] = "no_matches"
        return out

    top1 = matches[0]
    top_sim = float(top1.get("similarity", 0.0))
    out["top_sim"] = top_sim
    out["top1"] = top1

    if top_sim < threshold:
        out["note"] = "below_threshold"
        return out

    stats = get_cluster_stats(top1.get("sub_cluster"), top1.get("sub_name"), large_category)
    if stats is None:
        out["note"] = "cluster_missing"
        return out
    out["stats"] = stats

    lower = stats.get("lower_bound")
    upper = stats.get("upper_bound")
    need_info = bool(stats.get("need_info"))
    has_bounds = lower is not None and upper is not None

    # Price decides verdict — need_info only adds an extra warning.
    price = float(estimate_value)
    if has_bounds:
        lo, up = float(lower), float(upper)
        if lo <= price <= up:
            out["verdict"] = "ACCEPTABLE"
        elif price < lo:
            # Below the lower bound → review needed
            out["verdict"] = "REVIEW NEEDED"
            out["note"] = "too_low"
        else:
            # Above upper bound — compare to cluster max
            cluster_prices = fetch_cluster_prices(
                top1.get("sub_cluster"),
                top1.get("sub_name"),
                large_category,
            )
            cluster_max = max(cluster_prices) if cluster_prices else up
            out["cluster_max"] = cluster_max
            if price <= cluster_max:
                out["verdict"] = "REVIEW NEEDED"
                out["note"] = "above_upper_within_max"
            else:
                out["verdict"] = "UNAPPROVED"
                out["note"] = "above_max"
    else:
        # No bounds available — approve by default
        out["verdict"] = "ACCEPTABLE"

    # need_info → flag the cluster as uncertain, regardless of verdict.
    # If a "too_low" note already exists, keep it; otherwise use "need_info".
    if need_info and out["note"] is None:
        out["note"] = "need_info"
    out["need_info"] = need_info
    return out


# ------------------------- UI -------------------------

st.set_page_config(page_title="Approve Helper: Estimate Value Checker", layout="wide")
st.title("Approve Helper: Estimate Value Checker")
st.caption(
    "Enter multiple items at once. The system embeds each item_label, "
    "finds the closest historical cluster, and decides "
    "ACCEPTABLE / REVIEW NEEDED / UNAPPROVED / NO REFERENCE."
)

LARGE_CATEGORIES = ["ホール", "厨房（機器以外）"]

VERDICT_COLOR = {
    "ACCEPTABLE":     "#16a34a",  # green-600
    "REVIEW NEEDED": "#f59e0b",  # amber-500
    "UNAPPROVED": "#dc2626",  # red-600
    "NO REFERENCE":   "#6b7280",  # gray-500
}

# ---------------- Session state init ----------------

if "entries" not in st.session_state:
    st.session_state.entries = [{"item_label": "", "estimate_value": 0.0}]
if "results" not in st.session_state:
    st.session_state.results = None
if "expanded" not in st.session_state:
    st.session_state.expanded = set()

# ---------------- Search settings (top) ----------------

with st.expander("Search settings", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Top-1 hybrid score must be ≥ this to trust the verdict.",
        )
    with c2:
        vector_weight = st.slider(
            "Vector weight (semantic)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Keyword weight = 1 − vector weight.",
        )

# ---------------- Large category ----------------

large_category = st.selectbox(
    "Large category",
    options=LARGE_CATEGORIES,
    index=0,
    help="All items in this batch share the same large category.",
)

# ---------------- Items ----------------

st.subheader("Items")

estimation_title = st.text_input(
    "Estimation title",
    placeholder="e.g. Quote #12345 — Kitchen renovation",
    help="Free text for display only.",
)

remove_idx = None
for idx, item in enumerate(st.session_state.entries):
    rc_label, rc_price, rc_remove = st.columns([5, 5, 0.4])
    with rc_label:
        item["item_label"] = st.text_input(
            "Item label",
            value=item["item_label"],
            key=f"item_label_{idx}",
            placeholder="e.g. クロス補修",
            label_visibility="visible" if idx == 0 else "collapsed",
        )
    with rc_price:
        item["estimate_value"] = st.number_input(
            "Estimate value",
            min_value=0.0,
            value=float(item["estimate_value"]),
            step=100.0,
            format="%.2f",
            key=f"estimate_value_{idx}",
            label_visibility="visible" if idx == 0 else "collapsed",
        )
    with rc_remove:
        # spacer so the × button aligns vertically with the input box,
        # not with the input's label
        if idx == 0:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if len(st.session_state.entries) > 1:
            if st.button("✕", key=f"remove_{idx}", help="Remove this item"):
                remove_idx = idx

if remove_idx is not None:
    st.session_state.entries.pop(remove_idx)
    # clear results so stale display doesn't show
    st.session_state.results = None
    st.rerun()

# ---------------- Action buttons ----------------

if st.button("+ Add item", use_container_width=True):
    st.session_state.entries.append({"item_label": "", "estimate_value": 0.0})
    st.session_state.results = None
    st.rerun()

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

# ---------------- Analyze ----------------

if analyze_clicked:
    valid_items = [
        (i, it) for i, it in enumerate(st.session_state.entries)
        if it["item_label"].strip()
    ]
    if not valid_items:
        st.warning("Please enter at least one item label.")
    else:
        labels = [it["item_label"].strip() for _, it in valid_items]
        with st.spinner(f"Embedding {len(labels)} item(s)..."):
            embeddings = embed_batch(labels)

        progress = st.progress(0.0, text="Processing...")
        results: list[dict] = []
        for pos, ((orig_idx, it), emb) in enumerate(zip(valid_items, embeddings)):
            try:
                r = compute_verdict(
                    item_label=it["item_label"].strip(),
                    estimate_value=float(it["estimate_value"]),
                    large_category=large_category,
                    threshold=threshold,
                    vector_weight=vector_weight,
                    query_emb=emb,
                )
            except Exception as e:
                r = {
                    "verdict": "NO REFERENCE",
                    "note": f"error:{type(e).__name__}",
                    "matches": [],
                    "top_sim": None,
                    "stats": None,
                    "error": str(e),
                }
            r["_item_label"] = it["item_label"].strip()
            r["_estimate_value"] = float(it["estimate_value"])
            r["_large_category"] = large_category
            results.append(r)
            progress.progress((pos + 1) / len(valid_items),
                              text=f"Searched {pos + 1}/{len(valid_items)}")
        progress.empty()

        st.session_state.results = {
            "items":            results,
            "estimation_title": estimation_title,
            "large_category":   large_category,
            "threshold":        threshold,
            "vector_weight":    vector_weight,
        }
        st.session_state.expanded = set()

# ---------------- Render results ----------------

def render_badge(verdict: str) -> str:
    color = VERDICT_COLOR.get(verdict, "#6b7280")
    return (
        f'<div style="background:{color}15;border:2px solid {color};color:{color};'
        f'padding:6px 0;border-radius:8px;font-weight:800;text-align:center;'
        f'font-size:14px;letter-spacing:0.5px;">{verdict}</div>'
    )

def render_note(note: str | None) -> str:
    msg_map = {
        "too_low":                  "Price below historical range — please review",
        "above_upper_within_max":   "Price above expected range but within cluster max — please review",
        "above_max":                "Price exceeds the maximum seen in this cluster",
        "need_info":                "Cluster lacks information — range may be unreliable",
        "below_threshold":          "Top similarity below threshold — match is not reliable",
        "no_matches":               "No similar items found",
        "cluster_missing":          "Cluster stats missing for selected large_category",
    }
    if not note:
        return ""
    msg = msg_map.get(note, note)
    amber = {"too_low", "above_upper_within_max", "need_info"}
    if note in amber:
        color = "#b45309"
    elif note == "above_max":
        color = "#dc2626"
    else:
        color = "#6b7280"
    return f'<div style="font-size:12px;color:{color};margin-top:4px;">⚠ {msg}</div>'

def render_price_chart(
    sub_cluster: str,
    sub_name: str,
    large_category: str,
    stats: dict,
    estimate_value: float | None = None,
) -> None:
    """Histogram + KDE of cluster prices, with expected / lower / upper lines."""
    prices_list = fetch_cluster_prices(sub_cluster, sub_name, large_category)
    if not prices_list:
        st.info(
            "No `price_per_unit` rows found in `item_embeddings` for this cluster. "
            "Did you add the column and re-upload?"
        )
        return

    prices = np.array(prices_list, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=prices,
            histnorm="probability density",
            name="Histogram",
            opacity=0.5,
        )
    )

    max_y = 1.0
    if len(prices) > 1 and float(np.std(prices)) > 0:
        kde = gaussian_kde(prices)
        x_range = np.linspace(prices.min(), prices.max(), 200)
        y_kde = kde(x_range)
        max_y = float(y_kde.max())
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y_kde, mode="lines", name="KDE",
                line=dict(color="black", width=2),
            )
        )

    expected = stats.get("expected")
    lower    = stats.get("lower_bound")
    upper    = stats.get("upper_bound")

    if expected is not None:
        fig.add_trace(
            go.Scatter(
                x=[expected, expected], y=[0, max_y],
                mode="lines", name="expected (mean)",
                line=dict(color="green"),
            )
        )
    if lower is not None:
        fig.add_trace(
            go.Scatter(
                x=[lower, lower], y=[0, max_y],
                mode="lines", name="lower_bound",
                line=dict(color="red", dash="dash"),
            )
        )
    if upper is not None:
        fig.add_trace(
            go.Scatter(
                x=[upper, upper], y=[0, max_y],
                mode="lines", name="upper_bound",
                line=dict(color="blue", dash="dash"),
            )
        )

    # Mark user's estimate too — handy visual cue
    if estimate_value is not None and estimate_value > 0:
        fig.add_trace(
            go.Scatter(
                x=[estimate_value, estimate_value], y=[0, max_y],
                mode="lines", name="estimate value",
                line=dict(color="orange", width=3, dash="dot"),
            )
        )

    fig.update_layout(
        title=f"Price distribution — {sub_cluster} / {sub_name}  (n={len(prices)})",
        xaxis_title="Price per unit",
        yaxis_title="Density",
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_details(r: dict):
    """Inline details panel — shown when 'Show details' is toggled."""
    top1   = r.get("top1") or {}
    stats  = r.get("stats") or {}
    top_sim = r.get("top_sim")
    matches = r.get("matches") or []

    with st.container(border=True):
        st.markdown("**Best match cluster**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("sub_cluster", str(top1.get("sub_cluster", "-")))
        d2.metric("sub_name",    str(top1.get("sub_name",    "-")))
        d3.metric("Similarity",  f"{top_sim:.4f}" if top_sim is not None else "-")
        d4.metric("Samples used", stats.get("n_used", "-") if stats else "-")

        if stats:
            st.markdown("**Expected (cluster center)**")
            st.markdown(
                f"<div style='font-size:22px;font-weight:700;'>¥{fmt_money(stats.get('expected'))}</div>",
                unsafe_allow_html=True,
            )

        # --- Top 5 similar items (moved up) ---
        if matches:
            st.markdown(f"**Top {len(matches)} similar items**")
            st.dataframe(
                [
                    {
                        "rank":        i + 1,
                        "hybrid":      round(m.get("similarity", 0.0), 4),
                        "vector_sim":  round(m.get("vector_sim", 0.0), 4),
                        "keyword_sim": round(m.get("keyword_sim", 0.0), 4),
                        "item_label":  m.get("item_label"),
                        "sub_cluster": m.get("sub_cluster"),
                        "sub_name":    m.get("sub_name"),
                    }
                    for i, m in enumerate(matches)
                ],
                use_container_width=True,
                hide_index=True,
            )

        # --- Price distribution chart ---
        if stats and top1.get("sub_cluster") and top1.get("sub_name"):
            st.markdown("**Price distribution**")
            render_price_chart(
                sub_cluster=top1.get("sub_cluster"),
                sub_name=top1.get("sub_name"),
                large_category=r.get("_large_category"),
                stats=stats,
                estimate_value=r.get("_estimate_value"),
            )

            # --- All items in this cluster, FALSE rows highlighted red ---
            cluster_rows = fetch_cluster_items(
                top1.get("sub_cluster"),
                top1.get("sub_name"),
                r.get("_large_category"),
            )
            if cluster_rows:
                items_df = pd.DataFrame(cluster_rows)
                # Order columns for readability
                col_order = [c for c in
                             ["id", "item_label", "price_per_unit",
                              "is_in_hensachi_range", "trouble_id"]
                             if c in items_df.columns]
                items_df = items_df[col_order].sort_values(
                    by="price_per_unit",
                    ascending=True,
                    na_position="last",
                )

                def highlight_false(row):
                    if row.get("is_in_hensachi_range") is False:
                        return ["background-color: #fee2e2"] * len(row)  # red-100
                    return [""] * len(row)

                styled = items_df.style.apply(highlight_false, axis=1).format({
                    "price_per_unit": lambda v: "-" if pd.isna(v) else f"¥{float(v):,.0f}",
                })
                n_false = int((items_df["is_in_hensachi_range"] == False).sum()) \
                    if "is_in_hensachi_range" in items_df.columns else 0
                n_true  = int((items_df["is_in_hensachi_range"] == True).sum()) \
                    if "is_in_hensachi_range" in items_df.columns else 0
                st.markdown(
                    f"**All items in this cluster** "
                    f"(<span style='color:#16a34a'>TRUE: {n_true}</span> · "
                    f"<span style='color:#dc2626'>FALSE: {n_false}</span>)",
                    unsafe_allow_html=True,
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

if st.session_state.results:
    res = st.session_state.results
    st.divider()
    st.subheader("Results")
    if res.get("estimation_title"):
        st.markdown(
            f"**Estimation title:** {res['estimation_title']}  ·  "
            f"**Large category:** `{res['large_category']}`"
        )

    items = res["items"]

    # Per-item compact rows
    for i, r in enumerate(items):
        rc_btn, rc_label, rc_price, rc_range, rc_badge = st.columns([0.4, 2.6, 1.2, 1.6, 1.2])
        with rc_btn:
            expanded = i in st.session_state.expanded
            icon = "▼" if expanded else "▶"
            if st.button(icon, key=f"detail_btn_{i}", help="Show details"):
                if expanded:
                    st.session_state.expanded.discard(i)
                else:
                    st.session_state.expanded.add(i)
                st.rerun()
        with rc_label:
            st.markdown(f"**{r['_item_label']}**")
            st.markdown(render_note(r.get("note")), unsafe_allow_html=True)
            # Show need_info as an additional warning if it wasn't already the primary note
            if r.get("need_info") and r.get("note") != "need_info":
                st.markdown(render_note("need_info"), unsafe_allow_html=True)
        with rc_price:
            st.markdown(
                "<div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.5px;'>Estimate</div>"
                f"<div style='font-size:18px;font-weight:600;'>¥{r['_estimate_value']:,.0f}</div>",
                unsafe_allow_html=True,
            )
        with rc_range:
            stats = r.get("stats") or {}
            lb = stats.get("lower_bound")
            ub = stats.get("upper_bound")
            if lb is not None and ub is not None:
                range_text = f"¥{float(lb):,.0f}  ~  ¥{float(ub):,.0f}"
            else:
                range_text = "—"
            st.markdown(
                "<div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.5px;'>Price range</div>"
                f"<div style='font-size:15px;font-weight:500;'>{range_text}</div>",
                unsafe_allow_html=True,
            )
        with rc_badge:
            st.markdown(render_badge(r["verdict"]), unsafe_allow_html=True)
        if i in st.session_state.expanded:
            render_details(r)
        st.markdown("<hr style='margin:8px 0;border:0;border-top:1px solid #eee;'>",
                    unsafe_allow_html=True)
