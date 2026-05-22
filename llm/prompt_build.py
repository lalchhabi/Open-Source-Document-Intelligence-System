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
