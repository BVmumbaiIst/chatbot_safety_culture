import os
import sqlite3
import pandas as pd
import streamlit as st
import json
import numpy as np
import traceback
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate
import language_tool_python

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Database ---
from sqlalchemy import create_engine
from openai import OpenAI

# --- AWS + Utilities ---
import boto3
import io
import tempfile
import atexit
import time
from botocore.exceptions import ClientError
import uuid


DEBUG = False 
# ------------------------
# Load environment variables
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY not found in environment — LLM calls may fail.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------
# S3 configuration
# ------------------------
BUCKET_NAME = "iauditorsafetydata"
S3_KEYS = {
    "items": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_items.db",
    "users": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_users.db"
}

s3 = boto3.client("s3")

# ------------------------
# Helper: Clean old temp DB files (>24 hours)
# ------------------------
def cleanup_old_dbs(tmp_dir=tempfile.gettempdir(), hours=24):
    cutoff = time.time() - hours * 1800
    for file in os.listdir(tmp_dir):
        if file.endswith(".db"):
            path = os.path.join(tmp_dir, file)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except Exception:
                pass
# ------------------------
# Helper: Verify S3 object
# ------------------------
def verify_s3_file(bucket, key):
    """Check if the file exists in S3 and return size."""
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        size_mb = response["ContentLength"] / (1024 * 1024)
        return size_mb
    except ClientError as e:
        raise FileNotFoundError(f"File not found or no permission: {key}\n{e}")
# ------------------------
# Load SQLite safely from S3 (silent version)
# ------------------------
def load_sqlite_from_s3(s3_key: str):
    """Download SQLite DB safely from S3, ensuring complete file before use."""
    # Step 1: Verify S3 file
    head = s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
    size_mb = head["ContentLength"] / (1024 * 1024)

    # Step 2: Create unique local path
    unique_suffix = str(uuid.uuid4())[:8]
    local_path = os.path.join(
        tempfile.gettempdir(), f"{os.path.basename(s3_key).split('.')[0]}_{unique_suffix}.db"
    )

    # Step 3: Download with retry
    for attempt in range(3):
        try:
            if os.path.exists(local_path):
                os.remove(local_path)

            s3.download_file(BUCKET_NAME, s3_key, local_path)

            # Validate file
            if os.path.getsize(local_path) < 1024:
                raise ValueError("File too small — possibly incomplete")

            conn = sqlite3.connect(local_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            conn.close()

            if tables.empty:
                raise ValueError("No tables found — possibly corrupted")

            # Delete temp DB on exit
            atexit.register(lambda: os.path.exists(local_path) and os.remove(local_path))
            return local_path

        except Exception:
            time.sleep(2)

    raise RuntimeError(f"Failed to download and verify {s3_key} after multiple attempts.")

# ------------------------
# Safe DB connection + table detection
# ------------------------
def get_first_table_name(conn):
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    if tables.empty:
        raise ValueError("No tables found in the database.")
    return tables.iloc[0, 0]

def load_table_dynamic(db_path, expected_table_name=None):
    """Loads the table by expected name or auto-detects if not found."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        available = tables["name"].tolist()

        # Pick best match (case-insensitive)
        if expected_table_name:
            match = next((t for t in available if t.lower() == expected_table_name.lower()), None)
        else:
            match = None

        if not match:
            match = get_first_table_name(conn)

        df = pd.read_sql(f'SELECT * FROM "{match}"', conn)
        conn.close()
        return df

    except Exception as e:
        print(f"❌ Failed to load from {db_path}: {e}")
        return pd.DataFrame()

# ------------------------
# Download both DBs silently
# ------------------------
try:
    DB_PATH_ITEMS = load_sqlite_from_s3(S3_KEYS["items"])
    DB_PATH_USERS = load_sqlite_from_s3(S3_KEYS["users"])
except Exception as e:
    print(f"❌ Failed to load databases: {e}")
    DB_PATH_ITEMS = DB_PATH_USERS = None

# ------------------------
# Load both datasets
# ------------------------
df_items = load_table_dynamic(DB_PATH_ITEMS, "inspection_employee_schedule_items")
df_users = load_table_dynamic(DB_PATH_USERS, "inspection_employee_schedule_users")

# ------------------------
# Utility: Get single table name from SQLite DB
# ------------------------
def get_single_table_name(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    if not tables:
        raise ValueError("No tables found in the database.")
    return tables[0]

# ------------------------
# Database connections (assumes your S3 load already set paths)
# ------------------------
def get_connection_items():
    return sqlite3.connect(DB_PATH_ITEMS)

def get_connection_users():
    return sqlite3.connect(DB_PATH_USERS)
    
# ------------------------
# Extract filter options safely
# ------------------------
def get_filter_options_items(conn_items):
    table = get_single_table_name(conn_items)
    if DEBUG:
        st.info(f"Using items DB table: {table}")

    df = pd.read_sql(f'SELECT * FROM "{table}"', conn_items)

    if "date completed" in df.columns:
        df["date completed"] = pd.to_datetime(df["date completed"], errors="coerce")

    filters = {
        "date_min": df["date completed"].min() if "date completed" in df.columns else None,
        "date_max": df["date completed"].max() if "date completed" in df.columns else None,
        "regions": sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else [],
        "templates": sorted(df["TemplateNames"].dropna().unique().tolist()) if "TemplateNames" in df.columns else [],
        "employees": sorted(df["owner name"].dropna().unique().tolist()) if "owner name" in df.columns else [],
        "statuses": sorted(df["assignee status"].dropna().unique().tolist()) if "assignee status" in df.columns else [],
        "employeestatus": sorted(df["employee status"].dropna().unique().tolist()) if "employee status" in df.columns else []
    }

    return filters


def get_valid_emails(conn_users):
    table = get_single_table_name(conn_users)
    if DEBUG:
        st.info(f"Using users DB table: {table}")

    df_user = pd.read_sql(f'SELECT * FROM "{table}"', conn_users)
    return sorted(df_user["email"].dropna().unique().tolist()) if "email" in df_user.columns else []
    
# ------------------------
# Load and Validate Databases
# ------------------------
try:
    conn_items = get_connection_items()
    conn_users = get_connection_users()

    filters = get_filter_options_items(conn_items)
    emails = get_valid_emails(conn_users)

    # Clean UI message only
    st.success("✅ Data loaded successfully and chatbot is ready.")

except Exception as e:
    st.error(f"❌ Failed to load data: {e}")

finally:
    if 'conn_items' in locals():
        conn_items.close()
    if 'conn_users' in locals():
        conn_users.close()


# ------------------------
# LLM + Memory
# ------------------------
@st.cache_resource
def setup_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    return llm

# Initialize LLM
llm = setup_llm()

# ------------------------
# Streamlit UI basic config
# ------------------------
st.set_page_config(page_title="Interactive Data Chatbot", layout="wide")
st.title("💬 Interactive Data Chatbot + Analytics (Fixed)")

# If DB paths missing, stop
if DB_PATH_ITEMS is None or DB_PATH_USERS is None:
    st.stop()

# Create connections
conn_items = get_connection_items()
conn_users = get_connection_users()

# Get filter options and valid emails
try:
    filters_items = get_filter_options_items(conn_items)
except Exception as e:
    st.error(f"Failed to load filter options from items DB: {e}")
    st.stop()

valid_emails = get_valid_emails(conn_users)

# ------------------------
# Sidebar login & filters
# ------------------------
with st.sidebar:
    st.header("🔑 Login")
    entered_email = st.text_input("Enter your Email")
    if st.button("Login"):
        if entered_email:
            if entered_email in valid_emails:
                st.session_state["logged_in"] = True
                st.session_state["email"] = entered_email
                st.success(f"✅ Logged in as: {entered_email}")
            else:
                st.session_state["logged_in"] = False
                st.error("❌ Access denied. Email not found.")
        else:
            st.warning("Please enter an email.")

if not st.session_state.get("logged_in", False):
    st.warning("🔒 Please log in to access filters and data.")
    st.stop()

# ------------------------
# Sidebar Filters
# ------------------------
st.sidebar.header("🔎 Apply Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [filters_items["date_min"], filters_items["date_max"]] if filters_items["date_min"] is not None else [],
    min_value=filters_items["date_min"],
    max_value=filters_items["date_max"]
)
region = st.sidebar.multiselect("Select Regions", filters_items["regions"])
template = st.sidebar.multiselect("Select Template", filters_items["templates"])
employee = st.sidebar.multiselect("Select Employee (Owner Name)", filters_items["employees"])
status = st.sidebar.multiselect("Select Assignee Status", filters_items["statuses"])
employee_status = st.sidebar.multiselect("Select Employee Status", filters_items["employeestatus"])
row_limit = st.sidebar.slider("Limit number of rows:", min_value=10, max_value=5000, value=200, step=10)

# ------------------------
# Dynamic SQL Query Builder
# ------------------------
items_table_name = get_single_table_name(conn_items)

sql_filters = []

if date_range:
    start_date = pd.to_datetime(date_range[0]).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(date_range[1]).strftime("%Y-%m-%d")
    sql_filters.append(f'"date completed" BETWEEN "{start_date}" AND "{end_date}"')

if region:
    region_values = ",".join([f"'{r}'" for r in region])
    sql_filters.append(f"region IN ({region_values})")

if template:
    template_values = ",".join([f"'{t}'" for t in template])
    sql_filters.append(f'"TemplateNames" IN ({template_values})')

if employee:
    employee_values = ",".join([f"'{e}'" for e in employee])
    sql_filters.append(f'"owner name" IN ({employee_values})')

if status:
    status_values = ",".join([f"'{s}'" for s in status])
    sql_filters.append(f'"assignee status" IN ({status_values})')

if employee_status:
    employee_status_values = ",".join([f"'{es}'" for es in employee_status])
    sql_filters.append(f'"employee status" IN ({employee_status_values})')

where_clause = " AND ".join(sql_filters) if sql_filters else "1=1"
default_query = f'SELECT * FROM "{items_table_name}" WHERE {where_clause} LIMIT {row_limit};'
sql_query = st.sidebar.text_area("✏️ Edit SQL Query", value=default_query, height=140)
st.sidebar.code(sql_query, language="sql")


if st.sidebar.button("Run Query"):
    try:
        df = pd.read_sql(sql_query, conn_items)
        st.session_state["filtered_df"] = df
        st.success(f"Loaded {len(df)} rows.")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ SQL Error: {e}")

# ------------------------
# Setup Agents
# ------------------------
@st.cache_resource
def setup_sql_agent():
    # use local DB path (DB_PATH_ITEMS) for agent
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH_ITEMS}")
    llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    return create_sql_agent(llm=llm_local, db=db, agent_type="openai-tools", verbose=True)

@st.cache_resource
def setup_vector_rag():
    # Build vectorstore from items DB table
    engine = create_engine(f"sqlite:///{DB_PATH_ITEMS}")
    # detect table name again
    try:
        df = pd.read_sql(f'SELECT * FROM "{items_table_name}" LIMIT 20000', engine)
    except Exception:
        df = pd.read_sql(f'SELECT * FROM "{items_table_name}" LIMIT 2000', engine)  # fallback smaller
    docs = [Document(page_content=row.to_json(), metadata={"row": i}) for i, row in df.iterrows()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    vectorstore = FAISS.from_documents(chunks, embeddings) if embeddings else None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) if vectorstore else None
    llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    return RetrievalQA.from_chain_type(llm=llm_local, retriever=retriever, chain_type="stuff") if retriever else None

sql_agent = setup_sql_agent()
rag_chain = setup_vector_rag()

# ------------------------
# Only generate visuals / summaries if df exists
# ------------------------
if "filtered_df" in st.session_state:
    df = st.session_state["filtered_df"]


# ------------------------
# Chatbot Logic
# ------------------------
# ------------------------
# Helper: Summarize DataFrame
# ------------------------
def generate_dataframe_summary(df):
    """Return a textual summary for the LLM."""
    try:
        numeric_summary = df.describe(include=[np.number]).transpose().round(2)
        categorical_summary = {
            col: df[col].value_counts().head(5).to_dict()
            for col in df.select_dtypes(include='object').columns
        }
        summary_text = f"Numerical Summary:\n{numeric_summary.to_string()}\n\nTop 5 Categories per Column:\n{json.dumps(categorical_summary, indent=2)}"
        return summary_text
    except Exception as e:
        return f"⚠️ Error while summarizing DataFrame: {e}"


# ------------------------
# Smart Context Detector
# ------------------------
def detect_query_relevance(llm_model, df, user_query):
    """Ask the LLM whether the user_query is related to the filtered df preview."""
    if llm_model is None:
        # fallback heuristic: check if any column name or top values appear in query
        q = user_query.lower()
        for col in df.select_dtypes(include='object').columns:
            if col.lower() in q:
                return True
            top_vals = df[col].dropna().astype(str).unique()[:5]
            if any(str(v).lower() in q for v in top_vals):
                return True
        return False

    df_preview = df.head(5).to_dict(orient="records")
    prompt = f"""
You are an assistant. Here is a preview of the filtered dataset (first rows):
{json.dumps(df_preview, indent=2)}

The user's question:
"{user_query}"
Task: respond ONLY with RELATED or UNRELATED depending on whether the user's question is clearly about the data shown."""
    try:
        out = llm_model.invoke(prompt).strip().upper()
        return "RELATED" in out
    except Exception:
        # fallback heuristic
        return detect_query_relevance(None, df, user_query)




# ------------------------
# Generate Analytical Report
# ------------------------
def generate_report_with_insights(summary, question, llm_model, relevance):
    if llm_model is None:
        # simple fallback textual report
        if relevance:
            return f"(No LLM available) The filtered data summary:\n{summary}\nQuestion: {question}"
        else:
            return f"(No LLM available) Your question doesn't look related to the filtered data. Summary of filtered data:\n{summary}"

    ctx = (
        "The user's question is related to the filtered dataset."
        if relevance
        else "The question seems unrelated; provide a short summary of the filtered data instead."
    )

    prompt = f"""
You are a senior data analyst.

{ctx}

Filtered data summary:
{summary}

User question:
{question}

Requirements:
- If related -> answer using the filtered data and highlight top/bottom performers, anomalies, and 2 actionable recommendations.
- If unrelated -> politely state it's unrelated and provide a concise summary of the filtered data.
Keep it professional and concise.
"""

    try:
        response = llm_model.invoke(prompt)
        # ✅ Extract text properly (handles both LangChain and dict outputs)
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return str(response)
    except Exception as e:
        return f"❌ Error generating report: {e}"


# ------------------------
# Hybrid Logic: SQL + RAG
# ------------------------
def get_chatbot_response(user_query, sql_agent=None, rag_chain=None):
    sql_keywords = [
        "average", "sum", "top", "count", "max", "min", "group by", "trend",
        "between", "total", "where", "order by", "compare", "ratio", "percentage"
    ]
    q = user_query.lower()
    if any(k in q for k in sql_keywords) and sql_agent:
        try:
            response = sql_agent.invoke({"input": user_query})
            return response.get("output", "⚠️ No SQL result found.")
        except Exception as e:
            return f"❌ SQL Agent error: {e}"
    elif rag_chain:
        try:
            response = rag_chain.run(user_query)
            return response
        except Exception as e:
            return f"❌ RAG error: {e}"
    else:
        return "⚠️ No LLM or retriever available."


# ------------------------
# Visual Generator
# ------------------------
def auto_generate_visuals(df, user_query):
    """Automatically detect key columns and render visuals relevant to user's question."""
    st.markdown("### 📊 Auto-Generated Visual Insights")
    vivid_colors = px.colors.qualitative.Vivid

    try:
        if "region" in df.columns and "TemplateNames" in df.columns:
            region_count = df.groupby("region")["TemplateNames"].count().reset_index(name="count")
            fig = px.bar(region_count, x="region", y="count", text="count",
                         color="region", color_discrete_sequence=vivid_colors,
                         title="Inspections by Region")
            st.plotly_chart(fig, use_container_width=True)

        if "TemplateNames" in df.columns:
            template_count = df["TemplateNames"].value_counts().head(10).reset_index()
            template_count.columns = ["TemplateNames", "count"]
            fig = px.bar(template_count, x="TemplateNames", y="count",
                         color="TemplateNames", text="count",
                         title="Top 10 Templates by Inspection Count")
            st.plotly_chart(fig, use_container_width=True)

        if "owner name" in df.columns:
            emp_count = df["owner name"].value_counts().head(10).reset_index()
            emp_count.columns = ["owner name", "count"]
            fig = px.bar(emp_count, x="owner name", y="count",
                         color="owner name", text="count",
                         title="Top 10 Employees by Inspection Count")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not generate visuals automatically: {e}")



# ------------------------
# Streamlit Chat Layout
# ------------------------
col_left, col_right = st.columns([1, 0.6])

with col_left:
    st.subheader("💬 Ask a Question About the Data")
    user_question = st.text_input("Enter your question:")

    if st.button("Ask Chatbot"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                # ✅ CASE 1: filtered_df available
                if "filtered_df" in st.session_state and not st.session_state["filtered_df"].empty:
                    df = st.session_state["filtered_df"]
                    st.info("🔎 Analyzing filtered dataset...")

                    summary = generate_dataframe_summary(df)
                    relevance = detect_query_relevance(llm, df, user_question)

                    if relevance:
                        st.success("🧠 Query detected as related to filtered data.")
                    else:
                        st.warning("⚠️ Query seems unrelated — summarizing filtered data context.")

                    answer = generate_report_with_insights(summary, user_question, llm, relevance)
                    st.markdown("### 📋 Chatbot Response")
                    st.write(answer)
                    
                    if relevance:
                        auto_generate_visuals(df, user_question)

                # ✅ CASE 2: No filtered data → SQL + RAG hybrid
                else:
                    st.info("📚 Querying full dataset via hybrid SQL + RAG...")
                    final_answer = get_chatbot_response(user_question, sql_agent, rag_chain)
                    st.markdown("### 📋 Chatbot Response (Full Database)")
                    st.write(final_answer)

            except Exception as e:
                st.error(f"❌ Error: {e}")


# ------------------------
# Right: Visual on Right columns
# ------------------------
def apply_chart_theme(fig):
    """Apply a consistent transparent visual theme."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", size=12),
        title_x=0.05,
        title_font=dict(size=18),
        showlegend=True,
        margin=dict(l=30, r=30, t=60, b=40)
    )
    return fig


def chart_header(title, key):
    """Render chart header with title (left) and small dropdown (right)."""
    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(f"### {title}")

    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Pie Chart"],
            key=key,
            label_visibility="collapsed",
            index=0
        )

    return chart_type


def generate_visuals(df):
    """Generate aggregated visualizations for filtered dataframe."""
    visuals = {}
    vivid_colors = px.colors.qualitative.Vivid
    bold_colors = px.colors.qualitative.Bold

    # 🌐 Inspections per Region
    if {"region", "TemplateNames"}.issubset(df.columns):
        region_count = df.groupby("region")["TemplateNames"].count().reset_index(name="count")
        chart_type = chart_header("🌐 Inspections per Region", "region_chart_type")

        if chart_type == "Bar Chart":
            fig = px.bar(
                region_count, x="region", y="count", text="count",
                color="region", color_discrete_sequence=vivid_colors
            )
        else:
            fig = px.pie(
                region_count, names="region", values="count",
                color_discrete_sequence=vivid_colors
            )

        visuals["inspections_per_region"] = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True,key="region_chart")

    # 📋 Inspections per Template
    if {"TemplateNames", "owner name"}.issubset(df.columns):
        template_count = df.groupby("TemplateNames")["owner name"].count().reset_index(name="count")
        chart_type = chart_header("📋 Inspections per Template", "template_chart_type")

        if chart_type == "Bar Chart":
            fig = px.bar(
                template_count, x="TemplateNames", y="count", text="count",
                color="TemplateNames", color_discrete_sequence=bold_colors
            )
        else:
            fig = px.pie(
                template_count, names="TemplateNames", values="count",
                color_discrete_sequence=bold_colors
            )

        visuals["inspections_per_template"] = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True,key="template_chart")

    # 🧑‍💼 Inspections per Employee
    if "owner name" in df.columns:
        emp_count = df["owner name"].value_counts().reset_index()
        emp_count.columns = ["owner name", "count"]
        chart_type = chart_header("🧑‍💼 Inspections per Employee", "employee_chart_type")

        if chart_type == "Bar Chart":
            fig = px.bar(
                emp_count, x="owner name", y="count", text="count",
                color="owner name", color_discrete_sequence=vivid_colors
            )
        else:
            fig = px.pie(
                emp_count, names="owner name", values="count",
                color_discrete_sequence=vivid_colors
            )

        visuals["inspections_per_employee"] = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True,key="employee_chart")

    # ✅🚫⏳ Inspections per Assignee Status
    if "assignee status" in df.columns:
        status_count = df["assignee status"].value_counts().reset_index()
        status_count.columns = ["assignee status", "count"]
        chart_type = chart_header("✅🚫⏳ Inspections per Assignee Status", "assignee_chart_type")

        if chart_type == "Bar Chart":
            fig = px.bar(
                status_count, x="assignee status", y="count", text="count",
                color="assignee status", color_discrete_sequence=vivid_colors
            )
        else:
            fig = px.pie(
                status_count, names="assignee status", values="count",
                color_discrete_sequence=vivid_colors
            )

        visuals["status_counts"] = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True, key="assignee_chart")

    # 🏷️ Inspections per Response
    if "response" in df.columns:
        resp_count = df["response"].value_counts().reset_index()
        resp_count.columns = ["response", "count"]
        chart_type = chart_header("🏷️ Inspections per Response", "response_chart_type")

        if chart_type == "Bar Chart":
            fig = px.bar(
                resp_count, x="response", y="count", text="count",
                color="response", color_discrete_sequence=vivid_colors
            )
        else:
            fig = px.pie(
                resp_count, names="response", values="count",
                color_discrete_sequence=vivid_colors
            )

        visuals["response_counts"] = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True,key="response_chart")

    return visuals



# RIGHT: Data Visualizations
with col_right:
    st.subheader("📊 Filtered Data & Visualizations")

    # Check if a filtered dataframe exists
    if "filtered_df" in st.session_state and not st.session_state["filtered_df"].empty:
        df = st.session_state["filtered_df"]

        st.markdown("### 🔍 Filtered Data Table")
        st.dataframe(df, use_container_width=True)

        # ✅ Completion by month chart
        if "date completed" in df.columns and "TemplateNames" in df.columns:
            df = df.assign(
                completion_month=pd.to_datetime(df["date completed"], errors="coerce").dt.to_period("M").astype(str)
            )
            chart_df = (
                df.groupby("completion_month")["TemplateNames"]
                .count()
                .reset_index(name="template_count")
                .sort_values("completion_month")  # ensures chronological order
            )

            st.markdown("### 📅 Inspections by Completion Month")
            st.bar_chart(chart_df.set_index("completion_month")["template_count"])

        # ✅ Additional visuals
        visuals = generate_visuals(df)

    else:
        st.info("ℹ️ No data loaded yet. Please apply filters and click 'Run Query' first.")



