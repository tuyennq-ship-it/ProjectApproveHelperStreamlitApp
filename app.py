import os

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

CSV_PATH = os.path.join(os.path.dirname(__file__), "Top3_with_ info.csv")
HENSACHI_COL = "is_in_hensachi_range"
PRICE_COL = "Price per unit"


DROP_COLS = [
    "item_amount",
    "item_total",
    "item_quantity",
    "is_duplicate",
    "is_outlier",
    "is_same_prefecture",
    "distance_proxy",
    "remove",
    "recluster",
    "not exactly",
    "not_exactly",
]


@st.cache_data(show_spinner="Loading CSV…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "description_vi" in df.columns:
        df["description_vi"] = df["description_vi"].where(
            df["description_vi"].astype(str).str.strip() != "#VALUE!", None
        )
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if HENSACHI_COL in df.columns:
        df[HENSACHI_COL] = df[HENSACHI_COL].map(
            {True: True, False: False, "TRUE": True, "FALSE": False, "True": True, "False": False}
        )
    return df


def has_hensachi(group: pd.DataFrame) -> bool:
    if HENSACHI_COL not in group.columns:
        return False
    return group[HENSACHI_COL].notna().any()


def recompute_stats(prices: pd.Series) -> dict:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) == 0:
        return {"expected": np.nan, "std": np.nan, "lower": np.nan, "upper": np.nan, "n": 0}
    expected = float(prices.mean())
    std = float(prices.std(ddof=0)) if len(prices) > 1 else 0.0
    return {
        "expected": expected,
        "std": std,
        "lower": expected - std,
        "upper": expected + std,
        "n": int(len(prices)),
    }


def scroll_to_top() -> None:
    """Inject JS that scrolls the parent Streamlit page to the top."""
    components.html(
        """
        <script>
            const doc = window.parent.document;
            const main = doc.querySelector('section.main')
                      || doc.querySelector('[data-testid="stAppViewContainer"]')
                      || doc.scrollingElement
                      || doc.documentElement;
            if (main) main.scrollTo({top: 0, left: 0, behavior: 'auto'});
            window.parent.scrollTo({top: 0, left: 0, behavior: 'auto'});
        </script>
        """,
        height=0,
    )


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = df[df["need_info"] == False]  # noqa: E712
    for sub_cluster, g in eligible.groupby("sub_cluster", sort=False):
        sub_name = g["sub_name"].iloc[0]
        if has_hensachi(g):
            used = g[g[HENSACHI_COL] == True]  # noqa: E712
            stats = recompute_stats(used[PRICE_COL])
            is_exception = False
        else:
            stats = recompute_stats(g[PRICE_COL])
            is_exception = True
        rows.append(
            {
                "sub_cluster": sub_cluster,
                "sub_name": sub_name,
                "expected": stats["expected"],
                "std": stats["std"],
                "lower": stats["lower"],
                "upper": stats["upper"],
                "n_used": stats["n"],
                "n_total": int(len(g)),
                "is_exception": is_exception,
            }
        )
    return pd.DataFrame(rows)


def build_need_info_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pending = df[df["need_info"] == True]  # noqa: E712
    for sub_cluster, g in pending.groupby("sub_cluster", sort=False):
        sub_name = g["sub_name"].iloc[0]
        rows.append(
            {
                "sub_cluster": sub_cluster,
                "sub_name": sub_name,
                "n_total": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def render_range(row: pd.Series) -> str:
    if pd.isna(row["expected"]):
        return "—"
    return f"{fmt_money(row['expected'])} ± {fmt_money(row['std'])}  ({fmt_money(row['lower'])} → {fmt_money(row['upper'])})"


_DICT_CSS = """
<style>
.dict-card {
    border: 1px solid #e6e8eb;
    border-left: 4px solid #1f77b4;
    border-radius: 10px;
    padding: 10px 14px;
    background: #ffffff;
}
.dict-card.exception { border-left-color: #f0a000; background: #fffaf0; }
.dict-card.active {
    border-left-color: #2ca02c;
    background: #f3fbf5;
    box-shadow: 0 0 0 2px rgba(44,160,44,0.25);
}
.dict-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; margin-bottom: 6px;
}
.dict-title { font-size: 1.02rem; font-weight: 600; color: #1f2328; }
.dict-sc { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size: 0.78rem; color: #6a737d; background: #f3f4f6;
           padding: 2px 6px; border-radius: 6px; }
.dict-row { display: flex; flex-wrap: wrap; gap: 14px 22px; margin-top: 4px; }
.dict-kv { display: flex; flex-direction: column; }
.dict-k { font-size: 0.72rem; color: #6a737d; text-transform: uppercase; letter-spacing: .04em; }
.dict-v { font-size: 0.95rem; color: #1f2328; font-weight: 500; }
.dict-v.range { color: #0b5cad; font-variant-numeric: tabular-nums; }
.dict-flag { font-size: 0.78rem; padding: 2px 8px; border-radius: 999px; }
.dict-flag.ok { background: #e7f5ec; color: #1a7f37; }
.dict-flag.warn { background: #fff3d6; color: #9a6700; }
</style>
"""


def render_dictionary_cards(view: pd.DataFrame, highlight_sc: str | None = None) -> str | None:
    """Render dictionary as cards with a 'View details' button per item.

    Returns the sub_cluster whose button was clicked this run, or None.
    """
    st.markdown(_DICT_CSS, unsafe_allow_html=True)

    clicked: str | None = None
    for _, row in view.iterrows():
        is_exc = bool(row["is_exception"])
        is_active = (highlight_sc is not None and row["sub_cluster"] == highlight_sc)
        classes = ["dict-card"]
        if is_exc:
            classes.append("exception")
        if is_active:
            classes.append("active")

        flag_html = (
            '<span class="dict-flag warn">⚠️ Not Clean</span>'
            if is_exc else '<span class="dict-flag ok">✓ Clean</span>'
        )

        if pd.isna(row["expected"]):
            range_html = "—"
        else:
            range_html = (
                f'{fmt_money(row["expected"])} ± {fmt_money(row["std"])} '
                f'<span style="color:#6a737d">'
                f'({fmt_money(row["lower"])} → {fmt_money(row["upper"])})</span>'
            )

        sub_name = str(row["sub_name"]).replace("<", "&lt;").replace(">", "&gt;")
        sub_cluster = str(row["sub_cluster"]).replace("<", "&lt;").replace(">", "&gt;")

        card_html = (
            f'<div class="{" ".join(classes)}">'
            f'  <div class="dict-head">'
            f'    <div><span class="dict-title">{sub_name}</span> '
            f'         <span class="dict-sc">{sub_cluster}</span></div>'
            f'    {flag_html}'
            f'  </div>'
            f'  <div class="dict-row">'
            f'    <div class="dict-kv"><span class="dict-k">Price range</span>'
            f'         <span class="dict-v range">{range_html}</span></div>'
            f'    <div class="dict-kv"><span class="dict-k">Rows used</span>'
            f'         <span class="dict-v">{int(row["n_used"])} / {int(row["n_total"])}</span></div>'
            f'  </div>'
            f'</div>'
        )

        col_card, col_btn = st.columns([6, 1])
        with col_card:
            st.markdown(card_html, unsafe_allow_html=True)
        with col_btn:
            label = "✓ Viewing" if is_active else "View details"
            if st.button(
                label,
                key=f"viewbtn::{row['sub_cluster']}",
                disabled=is_active,
                use_container_width=True,
            ):
                clicked = str(row["sub_cluster"])
        st.write("")
    return clicked


def render_need_info_cards(view: pd.DataFrame, highlight_sc: str | None = None) -> str | None:
    """Render need_info sub_clusters as cards. No expected/std — just rows count."""
    st.markdown(_DICT_CSS, unsafe_allow_html=True)

    clicked: str | None = None
    for _, row in view.iterrows():
        is_active = (highlight_sc is not None and row["sub_cluster"] == highlight_sc)
        # Reuse 'exception' style for amber/yellow accent
        classes = ["dict-card", "exception"]
        if is_active:
            classes.append("active")

        sub_name = str(row["sub_name"]).replace("<", "&lt;").replace(">", "&gt;")
        sub_cluster = str(row["sub_cluster"]).replace("<", "&lt;").replace(">", "&gt;")

        card_html = (
            f'<div class="{" ".join(classes)}">'
            f'  <div class="dict-head">'
            f'    <div><span class="dict-title">{sub_name}</span> '
            f'         <span class="dict-sc">{sub_cluster}</span></div>'
            f'    <span class="dict-flag warn">need_info</span>'
            f'  </div>'
            f'  <div class="dict-row">'
            f'    <div class="dict-kv"><span class="dict-k">Status</span>'
            f'         <span class="dict-v">stats not computed</span></div>'
            f'    <div class="dict-kv"><span class="dict-k">Rows total</span>'
            f'         <span class="dict-v">{int(row["n_total"])}</span></div>'
            f'  </div>'
            f'</div>'
        )

        col_card, col_btn = st.columns([6, 1])
        with col_card:
            st.markdown(card_html, unsafe_allow_html=True)
        with col_btn:
            label = "✓ Viewing" if is_active else "View details"
            if st.button(
                label,
                key=f"viewbtn_ni::{row['sub_cluster']}",
                disabled=is_active,
                use_container_width=True,
            ):
                clicked = str(row["sub_cluster"])
        st.write("")
    return clicked


def need_info_detail_view(df: pd.DataFrame, sub_cluster: str) -> None:
    g = df[df["sub_cluster"] == sub_cluster].copy().reset_index(drop=True)
    sub_name = g["sub_name"].iloc[0] if len(g) else "—"

    st.markdown(f"### Detail: `{sub_cluster}` — **{sub_name}**")
    st.warning(
        "The meaning of the item label in this cluster is too vague to provide a price range."
        "Showing raw rows only."
    )
    st.caption(f"Rows total: **{len(g)}**")
    st.dataframe(
        g,
        hide_index=True,
        use_container_width=True,
        height=min(520, 60 + 35 * len(g)) if len(g) else 120,
    )


def detail_view(df: pd.DataFrame, sub_cluster: str) -> None:
    g = df[df["sub_cluster"] == sub_cluster].copy().reset_index(drop=True)
    sub_name = g["sub_name"].iloc[0]
    is_exception = not has_hensachi(g)

    st.markdown(f"### Detail: `{sub_cluster}` — **{sub_name}**")
    if is_exception:
        st.warning(
            "⚠️ The item labels in this cluster lack semantic similarity and need to be reviewed."
        )
        default_mask = pd.Series([True] * len(g))
    else:
        default_mask = g[HENSACHI_COL].fillna(False).astype(bool)

    state_key = f"mask::{sub_cluster}"
    if state_key not in st.session_state or len(st.session_state[state_key]) != len(g):
        st.session_state[state_key] = default_mask.tolist()

    colA, colB, _ = st.columns([1, 1, 4])
    with colA:
        if st.button("Reset to default", key=f"reset::{sub_cluster}"):
            st.session_state[state_key] = default_mask.tolist()
            st.rerun()
    with colB:
        if st.button("Select all / Clear all", key=f"toggle::{sub_cluster}"):
            cur = st.session_state[state_key]
            new_val = not all(cur)
            st.session_state[state_key] = [new_val] * len(g)
            st.rerun()

    edit_df = g.copy()
    edit_df.insert(0, "use", st.session_state[state_key])

    def _highlight_unused(row: pd.Series) -> list[str]:
        if not bool(row["use"]):
            return ["background-color: #ffe1e1"] * len(row)
        return [""] * len(row)

    styled_df = edit_df.style.apply(_highlight_unused, axis=1)

    edited = st.data_editor(
        styled_df,
        key=f"editor::{sub_cluster}",
        hide_index=True,
        column_config={
            "use": st.column_config.CheckboxColumn(
                "Use for stats",
                help="Tick = include this row in expected/std calculation. Unticked rows are highlighted red.",
                default=False,
            ),
        },
        disabled=[c for c in edit_df.columns if c != "use"],
        use_container_width=True,
        height=min(420, 60 + 35 * len(edit_df)),
    )
    st.session_state[state_key] = edited["use"].tolist()

    selected = edited[edited["use"] == True]  # noqa: E712
    stats = recompute_stats(selected[PRICE_COL])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows used", f"{stats['n']}/{len(g)}")
    m2.metric("Expected (mean)", fmt_money(stats["expected"]))
    m3.metric("Std", fmt_money(stats["std"]))
    m4.metric("Range ±std", f"{fmt_money(stats['lower'])} → {fmt_money(stats['upper'])}")

    if is_exception:
        return

    st.markdown("#### Price distribution (selected rows only)")
    prices = pd.to_numeric(selected[PRICE_COL], errors="coerce").dropna().values

    if len(prices) == 0:
        st.info("No selected rows with valid `Price per unit` to plot. Tick rows in **Use for stats**.")
        return
    if len(prices) < 2:
        st.info("Need at least 2 selected rows to plot distribution.")
        return

    fig = make_subplots(rows=1, cols=1, subplot_titles=(f"df_std_hensachi - {sub_cluster}",))
    fig.add_trace(
        go.Histogram(
            x=prices,
            histnorm="probability density",
            name="Histogram",
            opacity=0.5,
        ),
        row=1, col=1,
    )

    if np.std(prices) > 0:
        kde = gaussian_kde(prices)
        x_range = np.linspace(prices.min(), prices.max(), 100)
        kde_values = kde(x_range)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=kde_values, mode="lines", name="KDE",
                line=dict(color="black", width=2),
            ),
            row=1, col=1,
        )
        max_y = float(kde_values.max())
    else:
        max_y = 1.0

    if not pd.isna(stats["expected"]):
        fig.add_trace(
            go.Scatter(
                x=[stats["expected"], stats["expected"]], y=[0, max_y],
                mode="lines", name="expected (mean)",
                line=dict(color="green", dash="solid"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[stats["lower"], stats["lower"]], y=[0, max_y],
                mode="lines", name="lower_std",
                line=dict(color="red", dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[stats["upper"], stats["upper"]], y=[0, max_y],
                mode="lines", name="upper_std",
                line=dict(color="blue", dash="dash"),
            ),
            row=1, col=1,
        )

    fig.update_layout(
        height=500,
        title_text=f"Hensachi Analysis - {sub_cluster}",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Top3 sub_cluster explorer", layout="wide")
    st.title("Top3 sub_cluster — Expected / Std explorer")

    df = load_data(CSV_PATH)
    summary = build_summary(df)
    need_info_summary = build_need_info_summary(df)

    n_ok = int((~summary["is_exception"]).sum()) if len(summary) else 0
    n_exc = int(summary["is_exception"].sum()) if len(summary) else 0
    n_ni = len(need_info_summary)
    rows_ok = int(summary.loc[~summary["is_exception"], "n_total"].sum()) if len(summary) else 0
    rows_exc = int(summary.loc[summary["is_exception"], "n_total"].sum()) if len(summary) else 0
    rows_ni = int(need_info_summary["n_total"].sum()) if len(need_info_summary) else 0

    with st.sidebar:
        st.subheader("Filters")
        q = st.text_input("Search by sub_name / sub_cluster", "")
        st.caption(
            f"Clean: **{n_ok}** clusters / **{rows_ok}** rows  \n"
            f"Not Clean: **{n_exc}** clusters / **{rows_exc}** rows  \n"
            f"Need info: **{n_ni}** clusters / **{rows_ni}** rows"
        )

    def _filter(d: pd.DataFrame) -> pd.DataFrame:
        if not q.strip() or len(d) == 0:
            return d
        ql = q.strip().lower()
        return d[
            d["sub_name"].astype(str).str.lower().str.contains(ql)
            | d["sub_cluster"].astype(str).str.lower().str.contains(ql)
        ]

    ok_view = _filter(summary[summary["is_exception"] == False])  # noqa: E712
    exc_view = _filter(summary[summary["is_exception"] == True])  # noqa: E712
    ni_view = _filter(need_info_summary)

    valid_options = (
        set(ok_view["sub_cluster"].tolist())
        | set(exc_view["sub_cluster"].tolist())
        | set(ni_view["sub_cluster"].tolist())
    )
    current = st.session_state.get("chosen_sub_cluster")
    if current is not None and current not in valid_options:
        current = None
        st.session_state["chosen_sub_cluster"] = None

    if current is None:
        st.subheader("Dictionary")

        ok_rows = int(ok_view["n_total"].sum()) if len(ok_view) else 0
        exc_rows = int(exc_view["n_total"].sum()) if len(exc_view) else 0
        ni_rows = int(ni_view["n_total"].sum()) if len(ni_view) else 0

        mode_labels = {
            "ok": f"✓ Clean ({len(ok_view)} clusters / {ok_rows} rows)",
            "exception": f"⚠️ Not Clean ({len(exc_view)} clusters / {exc_rows} rows)",
            "need_info": f"🟡 Need info ({len(ni_view)} clusters / {ni_rows} rows)",
        }
        if "dict_mode" not in st.session_state:
            st.session_state["dict_mode"] = "ok"

        try:
            mode = st.segmented_control(
                "Mode",
                options=list(mode_labels.keys()),
                format_func=lambda k: mode_labels[k],
                default=st.session_state["dict_mode"],
                key="dict_mode_ctrl",
                label_visibility="collapsed",
            )
        except AttributeError:
            mode = st.radio(
                "Mode",
                options=list(mode_labels.keys()),
                format_func=lambda k: mode_labels[k],
                index=list(mode_labels.keys()).index(st.session_state["dict_mode"]),
                horizontal=True,
                key="dict_mode_ctrl",
                label_visibility="collapsed",
            )
        if mode is None:
            mode = st.session_state["dict_mode"]
        st.session_state["dict_mode"] = mode

        if mode == "ok":
            st.caption(
                "Range is recomputed from rows where `is_in_hensachi_range = TRUE`. "
                "Click **View details** on any item to inspect it."
            )
            if len(ok_view) == 0:
                st.info("No Clean sub_cluster matches the current filter.")
            else:
                clicked = render_dictionary_cards(ok_view, None)
                if clicked is not None:
                    st.session_state["chosen_sub_cluster"] = clicked
                    st.session_state["scroll_top"] = True
                    st.rerun()
        elif mode == "exception":
            st.caption(
                "The item labels in those clusters lack semantic similarity and need to be reviewed."
            )
            if len(exc_view) == 0:
                st.info("No Not Clean sub_cluster matches the current filter.")
            else:
                clicked = render_dictionary_cards(exc_view, None)
                if clicked is not None:
                    st.session_state["chosen_sub_cluster"] = clicked
                    st.session_state["scroll_top"] = True
                    st.rerun()
        else:  # need_info
            st.caption(
                "These clusters have too ambiguous semantics; it's not certain they belong to the same group for evaluation purposes."
                "Expected/std are **not** computed — only raw rows are shown on view details."
            )
            if len(ni_view) == 0:
                st.info("No need_info sub_cluster matches the current filter.")
            else:
                clicked_ni = render_need_info_cards(ni_view, None)
                if clicked_ni is not None:
                    st.session_state["chosen_sub_cluster"] = clicked_ni
                    st.session_state["scroll_top"] = True
                    st.rerun()
    else:
        if st.button("← Back to dictionary", key="back_to_dict"):
            st.session_state["chosen_sub_cluster"] = None
            st.session_state["scroll_top"] = True
            st.rerun()
        is_need_info = current in set(need_info_summary["sub_cluster"].tolist())
        if is_need_info:
            need_info_detail_view(df, current)
        else:
            detail_view(df, current)

    if st.session_state.pop("scroll_top", False):
        scroll_to_top()


if __name__ == "__main__":
    main()
