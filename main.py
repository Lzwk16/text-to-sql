import streamlit as st
import yaml

from src.text2sql import SQLPromptTemplate

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

db_path = config["db_path"]
template = config["prompt_template"]

query = st.text_area("Describe the data you want to retrieve from the database:")
sql_prompt = SQLPromptTemplate(model_name="deepseek-r1:8b", query=query)

schema = sql_prompt.extract_schema(db_path)
if query:
    sql = sql_prompt.text_to_query(schema, template)
    st.code(sql, wrap_lines=True, language="sql")
