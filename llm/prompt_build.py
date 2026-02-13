def build_prompt(retrieved_chunks, query, chat_history=None):
    # -----------------------
    # Build Context
    # -----------------------
    context = "\n\n".join(
        [f"- {chunk['text']}" for chunk in retrieved_chunks]
    )

    # -----------------------
    # Build Chat History
    # -----------------------
    history_text = ""

    if chat_history:
        for turn in chat_history[-3:]:  # last 3 turns only
            history_text += f"User: {turn['user']}\n"
            history_text += f"Assistant: {turn['assistant']}\n"

    # -----------------------
    # Final Prompt
    # -----------------------
    prompt = f"""
You are a helpful assistant.

Use BOTH the conversation history and context to answer.

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Instructions:
- Answer using ONLY the context
- Be concise and clear
- If not found, say "I don't know"

Answer:
"""

    return prompt
