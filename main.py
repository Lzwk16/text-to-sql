import streamlit as st

from src.text2sql import SQLPromptTemplate

db_path = "db/noshow.db"

template = """
You are a SQL generator.  When given a schema and a user question, you MUST output only the SQL statement—nothing else.  No explanation is needed.

Schema: {schema}
User question: {query}
Output (SQL only):
"""

query = st.text_area("Describe the data you want to retrieve from the database:")
sql_prompt = SQLPromptTemplate(model_name="deepseek-r1:8b", query=query)

schema = sql_prompt.extract_schema(db_path)
if query:
    sql = sql_prompt.text_to_query(schema, template)
    st.code(sql, wrap_lines=True, language="sql")
