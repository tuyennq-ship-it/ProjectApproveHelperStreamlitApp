import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = "text-embedding-3-small"

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )

    return response.data[0].embedding

print(embed("現場諸経費")[:10])

