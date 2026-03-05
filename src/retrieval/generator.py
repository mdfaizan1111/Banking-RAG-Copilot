# src/retrieval/generator.py
from __future__ import annotations
import os
from openai import OpenAI


def generate_answer(query: str, instructions: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        f"{instructions}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()