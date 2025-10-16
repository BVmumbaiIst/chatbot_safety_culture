import streamlit as st
import mysql.connector
from openai import OpenAI
from datetime import datetime

# ---- Securely load secrets ----
DB_HOST = st.secrets["DB_HOST"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_NAME = st.secrets["DB_NAME"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ---- MySQL connection ----
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def verify_user(email):
    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)  # ✅ ensures all results are read
    cursor.execute(
        "SELECT email FROM inspection_employee_schedule WHERE email = %s",
        (email,)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

def log_chat(email, user_message, bot_response):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chatbot_logs (email, user_message, bot_response, created_at)
        VALUES (%s, %s, %s, %s)
        """,
        (email, user_message, bot_response, datetime.now())
    )
    conn.commit()
    cursor.close()
    conn.close()

def fetch_chat_history(email, limit=20):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)
    cursor.execute(
        """
        SELECT user_message, bot_response, created_at
        FROM chatbot_logs
        WHERE email = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (email, limit)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# ---- Streamlit UI ----
st.title("💬 Safety Optimise Chatbot")

if "verified" not in st.session_state:
    email = st.text_input("Enter your registered email to access the chatbot:")

    if st.button("Verify Email"):
        if verify_user(email):
            st.session_state.verified = True
            st.session_state.email = email
            st.success("✅ Access granted! Welcome.")
        else:
            st.error("🚫 Access denied. Email not found in records.")

else:
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Sidebar chat history viewer
    st.sidebar.header("🗂️ Chat History")
    with st.sidebar:
        if st.button("🔄 Refresh History"):
            st.session_state.history = fetch_chat_history(st.session_state.email)

        if "history" not in st.session_state:
            st.session_state.history = fetch_chat_history(st.session_state.email)

        if len(st.session_state.history) == 0:
            st.sidebar.write("No chat history yet.")
        else:
            for chat in st.session_state.history:
                st.markdown(f"**🕒 {chat['created_at'].strftime('%Y-%m-%d %H:%M:%S')}**")
                st.markdown(f"**You:** {chat['user_message']}")
                st.markdown(f"**Bot:** {chat['bot_response']}")
                st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your safety-related question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate OpenAI response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Log chat to MySQL
        try:
            log_chat(st.session_state.email, prompt, response)
        except Exception as e:
            st.warning(f"⚠️ Could not log chat: {e}")

 # Step 2: Chatbot Interface
    client = OpenAI(api_key=OPENAI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your safety-related question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get OpenAI response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
