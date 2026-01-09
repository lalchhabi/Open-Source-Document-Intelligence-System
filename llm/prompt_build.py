def build_prompt(retrieved_chunks, query):
    context = "\n\n".join(
        [f"-{chunk['text']}" for chunk in retrieved_chunks]
    )
    prompt = f"""
    You are a helful assistant.
    Answer the question ONLY using the context below.
    Do NOT use external knowledge.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    print(f"Final prompt {prompt}")
    return prompt