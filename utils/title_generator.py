from database.db import get_session_messages, update_session_title, get_connection

from llm.prompt_build import generate_chat_title

def finalize_title(session_id, llm):
    # Load message from database
    messages = get_session_messages(session_id)

    # Don't generate title for very short charts
    if len(messages) < 4:
        return

    conn = get_connection()
    cursor = conn.cursor()

    # check current title
    cursor.execute("""
    SELECT title
    FROM chat_sessions
    WHERE id = ?
    """, (session_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return
    
    current_title = row['title']

    # only generate if title is default
    if current_title == "New Chat":
        title = generate_chat_title(
            llm,
            messages
        )
        update_session_title(
            session_id,
            title
        )
    conn.close()