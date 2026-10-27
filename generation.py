from ollama import chat

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the provided context. "
    "Rules:\n"
    "1. Use only facts found in the context; do not rely on outside knowledge.\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided context.\"\n"
    "3. Be concise and cite the supporting excerpt numbers like [1], [2].\n"
    "4. If excerpts conflict, point out the conflict instead of picking one silently."
)


def generate_answer(
    question: str,
    top_chunks: list[str],
    model: str = "llama3.2:3b",
    max_context_chars: int = 6000,
) -> str:
    # Clean: strip whitespace, drop empties and duplicates (keep order)
    seen, cleaned = set(), []
    for chunk in top_chunks:
        c = chunk.strip()
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)

    if not cleaned:
        return "I don't know based on the provided context."

    # Number the chunks and cap total size
    parts, total = [], 0
    for i, c in enumerate(cleaned, 1):
        entry = f"[{i}] {c}"
        if total + len(entry) > max_context_chars:
            break
        parts.append(entry)
        total += len(entry)
    context = "\n\n".join(parts)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (based only on the context above):"
    )

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()