# import libraries
import sqlite3
import uuid

# database path
DB_PATH = "database/chatbot.db"

def get_connection():
    """Function to connect the database
    """
    conn = sqlite3.connect(DB_PATH)

    # return rows like dictionary
    conn.row_factory = sqlite3.Row

    return conn


def init_db():
    """Functions to create tables for sessions and messages
    """

    conn = get_connection()

    cursor = conn.cursor()

    # chat session table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions(
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
    )
    """)

    # Messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN key(session_id)
        REFERENCES chat_sessions(id)
                   
        )
    """)

    conn.commit()
    conn.close()


def create_session():
    """Function to create the session
    """
    conn = get_connection()
    cursor = conn.cursor()
    session_id = str(uuid.uuid4())

    cursor.execute("""
        INSERT INTO chat_sessions (id, title)
        VALUES(?, ?)
    """, (session_id, "New Chat"))

    conn.commit()
    conn.close()

    return session_id

def get_all_sessions():
    """Load the sessions from the database
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM chat_sessions 
        ORDER BY created_at DESC
    """)

    sessions = cursor.fetchall()

    conn.close()

    return [dict(s) for s in sessions]

def save_message(session_id: int, role: str, content: str)->None:
    """Function to save message in the database

    Args:
        session_id (int): Unique session id
        role (str): Type of role: user or assistant
        content (str): message
    """
    if not session_id or not role or content is None:
        print("Skipping DB save due to missing values")
        return
    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO messages(
        session_id,
        role,
        content
    )
    VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()
    conn.close()

def get_session_messages(session_id: int)-> None:
    """Function to load session messages

    Args:
        session_id (int): Unique id of the session
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
    """
    SELECT role, content 
    FROM messages
    WHERE session_id = ?
    ORDER BY created_at ASC
    """, (session_id,))

    messages = cursor.fetchall()
    conn.close()

    return [dict(m) for m in messages]


def update_session_title(session_id, title):
    """Function that update session title

    Args:
        session_id (int): Unique session identity
        title (str): title of the session
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE chat_sessions
    SET title = ?
    WHERE id = ?
    """, (title, session_id))

    conn.commit()
    conn.close()


def session_exists(session_id):
    """Function to check whether session exist or not
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id
    FROM chat_sessions
    WHERE id = ?
    """, (session_id,))

    row = cursor.fetchone()

    conn.close()

    return row is not None

def get_session(session_id):
    
    """Function to load session
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM chat_sessions
    WHERE id = ?
    """, (session_id,))

    session = cursor.fetchone()

    conn.close()

    return dict(session) if session else None

def delete_session(session_id:int)->None:
    """Function that delete the specific selected session

    Args:
        session_id (int): Unique session identity number
    """
    conn = get_connection()
    cursor = conn.cursor()

    # delete messages first (important due to foreign relation logic)
    cursor.execute("""
        DELETE FROM messages
        WHERE session_id = ?
    """,(session_id,))

    # delete session
    cursor.execute("""
    DELETE FROM chat_sessions
    WHERE id = ?
    """, (session_id,))

    conn.commit()
    conn.close()