import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

# ----------------------------------------
# ⚙️ App Config
# ----------------------------------------
st.set_page_config(page_title="🦺 Safety Optimise Chatbot", layout="wide")
st.title("🦺 Safety Optimise Chatbot")
st.write("Chat securely with your inspection data using your company email access.")

# ----------------------------------------
# 🔐 Load Secrets from Streamlit
# ----------------------------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]
db_config = st.secrets["mysql"]

# Build MySQL connection string
DB_URI = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"

# Create SQLAlchemy engine
engine = create_engine(DB_URI)

# LangChain Database Wrapper
db = SQLDatabase(engine=engine)

# Initialize OpenAI Model (via LangChain)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-turbo", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# ----------------------------------------
# 🧠 Helper Functions
# ----------------------------------------
@st.cache_data(ttl=600)
def run_query(query: str):
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

def verify_user(email: str) -> bool:
    query = f"SELECT email FROM inspection_employee_schedule WHERE email = '{email}' LIMIT 1;"
    df = run_query(query)
    return not df.empty

# ----------------------------------------
# 🔑 Email Authentication
# ----------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    email_input = st.text_input("📧 Enter your company email to access the chatbot")

    if st.button("Verify Email"):
        if email_input:
            if verify_user(email_input):
                st.session_state.authenticated = True
                st.session_state.user_email = email_input
                st.success(f"✅ Access granted. Welcome, {email_input}!")
            else:
                st.error("🚫 Email not found in access list.")
        else:
            st.warning("Please enter a valid email address to proceed.")

# ----------------------------------------
# 💬 Chatbot Section
# ----------------------------------------
if st.session_state.authenticated:
    st.divider()
    st.subheader("🤖 Chat with your inspection database")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_prompt := st.chat_input("Ask me about inspections, employees, or schedules..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)

        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Generate SQL answer
        try:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Thinking..."):
                    result = agent_executor.run(user_prompt)

                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        response_text = f"Returned {len(result)} rows."
                    else:
                        response_text = result

                    st.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
