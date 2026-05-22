### Import libraries
from langchain_core.prompts import ChatPromptTemplate

def build_prompt():
    """Create a Langchain prompt template for the RAG pipeline this prompt is responsible got guiding the LLM to generate context-aware response.
    """
    prompt = ChatPromptTemplate.from_messages(
         [
             ### system message -> defines assistant role and behavior
            (
            "system",
    """
    You are a helpful AI assistant for document question answering. 
    Your task is to answer questions ONLY using the provided context.

    Rules:
    - Use the retrieved context as the primary source of truth
    - Use conversation history for continuity
    - If the answer is not found in the context, say:
    "This question does not appear related to the uploaded document.
    Please delete the uploaded document to switch back to normal chat mode."
    - Do not hallucinate or make up information
    - Keep answers clear and concise
    """
         ),

         ### Human message -> dynamic RAG input
         (
             "human",
    """
    Conversation History:
    {chat_history}

    Retrieved Context:
    {context}

    Question:
    {question}

    Answer:
    """
            )
         
        ]

    )
    return prompt


def generate_chat_title(llm, messages):
    """Generate a meaningful title from conversation history context with prompt
    """
    conversation_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages]

    )

    prompt = f"""
    You are a chat title generator.
    Create a short 3-6 word title for this conversation:

    {conversation_text}

    Rules:
    - No punctuation
    - No quotes
    - Very concise
    """
    response = llm.invoke(prompt)
    return response.content.strip()
