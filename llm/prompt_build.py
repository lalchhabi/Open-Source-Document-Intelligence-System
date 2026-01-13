def build_prompt(retrieved_chunks, query):
    context = "\n\n".join(
        [f"-{chunk['text']}" for chunk in retrieved_chunks]
    )
    prompt = f"""You are a helpful assistant.
    Answer the question using ONLY the information in the context below.
    Summarize the relevant details clearly and concisely.
    Do NOT use external knowledge.
    If the context does not contain relevant information, say "I don't know".

    Context:
    {context}

    Question:
    {query}

    Answer:"""

    print(f"Final prompt {prompt}")
    return prompt