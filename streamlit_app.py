import os
import sqlite3
import pandas as pd
import streamlit as st
import json
import re

import numpy as np
import traceback
from datetime import datetime
from dotenv import load_dotenv
from pandasql import sqldf
from tabulate import tabulate
import language_tool_python  # for grammar check

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
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
import tempfile
from io import BytesIO


# ------------------------
# Load environment variables
# ------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load environment variables (AWS credentials and OpenAI key)
load_dotenv()

# S3 bucket details
s3 = boto3.client("s3")
BUCKET_NAME = "iauditorsafetydata"
S3_KEYS = {
    "items": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_items.db",
    "users": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule.db"
}

def download_db_from_s3(s3_key):
    # Create a truly unique temp file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    local_path = tmp_file.name
    tmp_file.close()  # close it so Windows releases the lock
    s3.download_file(BUCKET_NAME, s3_key, local_path)
    return local_path

# Download
DB_PATH_ITEMS = download_db_from_s3(S3_KEYS["items"])
DB_PATH_USERS = download_db_from_s3(S3_KEYS["users"])

@st.cache_resource(ttl=3600)
def get_connection_items():
    return sqlite3.connect(DB_PATH_ITEMS, check_same_thread=False)

@st.cache_resource(ttl=3600)
def get_connection_users():
    return sqlite3.connect(DB_PATH_USERS, check_same_thread=False)
    
# ------------------------
# Get Filter Options
# ------------------------
def get_filter_options_items(conn_items):
    df = pd.read_sql("SELECT * FROM inspection_employee_schedule_items", conn_items)
    df["date completed"] = pd.to_datetime(df["date completed"], errors="coerce")
    return {
        "date_min": df["date completed"].min(),
        "date_max": df["date completed"].max(),
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "templates": sorted(df["TemplateNames"].dropna().unique().tolist()),
        "employees": sorted(df["owner name"].dropna().unique().tolist()),
        "statuses": sorted(df["assignee status"].dropna().unique().tolist()),
        "employeestatus": sorted(df["employee status"].dropna().unique().tolist())
    }


def get_valid_emails(conn_users):
    df_user = pd.read_sql("SELECT * FROM inspection_employee_schedule", conn_users)
    return sorted(df_user["email"].dropna().unique().tolist())


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
# Streamlit Config
# ------------------------
st.set_page_config(page_title="Interactive Data Chatbot", layout="wide")
st.title("💬 Interactive Data Chatbot + Analytics")

conn_items = get_connection_items()
conn_users = get_connection_users()
filters_items = get_filter_options_items(conn_items)
valid_emails = get_valid_emails(conn_users)

# ------------------------
# Sidebar Login
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
    [filters_items["date_min"], filters_items["date_max"]],
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
sql_filters = []

# Date range filter
if date_range:
    start_date = pd.to_datetime(date_range[0]).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(date_range[1]).strftime("%Y-%m-%d")
    sql_filters.append(f'"date completed" BETWEEN "{start_date}" AND "{end_date}"')

# Other filters
if region:
    region_list = ", ".join([f"'{r}'" for r in region])
    sql_filters.append(f"region IN ({region_list})")
if template:
    template_list = ", ".join([f"'{t}'" for t in template])
    sql_filters.append(f'"TemplateNames" IN ({template_list})')
if employee:
    employee_list = ", ".join([f"'{e}'" for e in employee])
    sql_filters.append(f'"owner name" IN ({employee_list})')
if status:
    status_list = ", ".join([f"'{s}'" for s in status])
    sql_filters.append(f'"assignee status" IN ({status_list})')
if employee_status:
    employee_status_list = ", ".join([f"'{es}'" for es in employee_status])
    sql_filters.append(f'"employee status" IN ({employee_status_list})')


where_clause = " AND ".join(sql_filters) if sql_filters else "1=1"
default_query = f'SELECT * FROM inspection_employee_schedule_items WHERE {where_clause} LIMIT {row_limit};'
sql_query = st.sidebar.text_area("✏️ Edit SQL Query", value=default_query, height=120)
st.sidebar.markdown("### 🔍 Final SQL Query")
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
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH_ITEMS}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    return create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)


@st.cache_resource
def setup_vector_rag():
    """Setup FAISS retriever once and reuse."""
    engine = create_engine(f"sqlite:///{DB_PATH_ITEMS}")
    df = pd.read_sql("SELECT * FROM inspection_employee_schedule_items LIMIT 20000", engine)

    docs = [Document(page_content=row.to_json(), metadata={"row": i}) for i, row in df.iterrows()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


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
    """Generate descriptive summary from filtered DataFrame."""
    try:
        numeric_summary = df.describe(include=[np.number]).transpose().round(2)
        categorical_summary = {
            col: df[col].value_counts().head(5).to_dict()
            for col in df.select_dtypes(include='object').columns
        }

        summary_text = f"""
        Numerical Summary:
        {numeric_summary.to_string()}

        Top 5 Categories per Column:
        {json.dumps(categorical_summary, indent=2)}
        """
        return summary_text

    except Exception as e:
        return f"⚠️ Error while summarizing DataFrame: {e}"


# ------------------------
# Smart Context Detector
# ------------------------
def detect_query_relevance(llm, df, user_query):
    """
    Use LLM to decide if the question is relevant to the filtered dataset.
    Returns True if related, False if unrelated.
    """
    df_preview = df.head(5).to_dict(orient="records")
    prompt = f"""
    You are an intelligent data analyst.

    Here's a preview of the filtered dataset (first few rows):
    {json.dumps(df_preview, indent=2)}

    The user's question is:
    "{user_query}"

    Task:
    - Determine if the user's question is clearly related to this filtered dataset.
    - For example:
        * If the question asks about columns, metrics, or values visible in this dataset → it's RELATED.
        * If it asks about something outside the dataset (e.g., different region, global summary, templates not in this subset) → it's UNRELATED.

    Answer ONLY with:
    "RELATED" or "UNRELATED"
    """
    result = llm.predict(prompt).strip().upper()
    return "RELATED" in result


# ------------------------
# Generate Analytical Report
# ------------------------
def generate_report_with_insights(summary, question, llm, relevance):
    """
    Generate a professional analytical report using LLM.
    """
    if relevance:
        context_instruction = "The user's question is related to the filtered dataset."
    else:
        context_instruction = "The question seems unrelated to the filtered dataset. Provide context summary instead."

    prompt = f"""
    You are a senior data analyst working on safety inspection data.

    {context_instruction}

    Filtered Data Summary:
    {summary}

    User Question:
    {question}

    Instructions:
    - If the question is related → answer it using the filtered data insights.
    - If it's unrelated → say so politely, then summarize what this filtered data represents.
    - Always include actionable insights and trends if possible.
    """

    return llm.predict(prompt)


# ------------------------
# Hybrid Logic: SQL + RAG
# ------------------------
def get_chatbot_response(user_query, sql_agent, rag_chain):
    """Use SQL Agent for analytical queries, RAG for general questions."""
    sql_keywords = [
        "average", "sum", "top", "count", "max", "min", "group by", "trend",
        "between", "total", "where", "order by", "compare", "ratio", "percentage"
    ]

    if any(k in user_query.lower() for k in sql_keywords):
        response = sql_agent.invoke({"input": user_query})
        return response.get("output", "⚠️ No SQL result found.")
    else:
        return rag_chain.run(user_query)


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

