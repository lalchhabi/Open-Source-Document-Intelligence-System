def build_prompt(retrieved_chunks, query, chat_history=None):
    """
    Build the final prompt for the LLM.

    This function combines:
    1. Retrieved document chunks (context)
    2. User query
    3. Optional chat history

    The final prompt is structured to guide the LLM to:
    - Answer based only on provided context
    - Use conversation history for continuity
    - Avoid hallucination

    Parameters
    ----------
    retrieved_chunks : list of dict
        List of relevant chunks retrieved from vector store.
        Each chunk contains:
        - "text": chunk content
        - "metadata": source info

    query : str
        User's current question.

    chat_history : list, optional
        Previous conversation history.
        Format:
        [
            {"user": "...", "assistant": "..."},
            ...
        ]

    Returns
    -------
    str
        Final formatted prompt ready for LLM input.

    Process:
    --------
    1. Combine retrieved chunks into a single context block
    2. Format recent chat history (last few turns)
    3. Construct structured prompt with instructions
    """

    # -----------------------
    # Step 1: Build Context
    # -----------------------
    # Combine all retrieved chunks into one text block
    # Each chunk is prefixed with "-" for readability
    context = "\n\n".join(
        [f"- {chunk['text']}" for chunk in retrieved_chunks]
    )

    # -----------------------
    # Step 2: Build Chat History
    # -----------------------
    history_text = ""

    # Include only recent history to avoid token overflow
    if chat_history:
        for turn in chat_history[-3:]:  # last 3 conversations only
            history_text += f"User: {turn['user']}\n"
            history_text += f"Assistant: {turn['assistant']}\n"

    # -----------------------
    # Step 3: Construct Final Prompt
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