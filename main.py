import os
import tempfile

import streamlit as st
import yaml

from src.text2sql import SQLPromptTemplate

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

db_path = config["db_path"]
template = config["prompt_template"]
model = config["local_llm"]

st.title("Natural Language to SQL Query")
st.write("Ask questions about your data in plain English!")

# Add database file uploader
uploaded_db = st.file_uploader(
    "Upload a SQLite database file (optional)", type=["db", "sqlite", "sqlite3"]
)
if uploaded_db:
    # Save the uploaded file temporarily

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(uploaded_db.getvalue())
        db_path = tmp_file.name

# Input for the query
query = st.text_area("What would you like to know about the data?")
sql_prompt = SQLPromptTemplate(model_name=model, query=query)

try:
    schema = sql_prompt.extract_schema(db_path)
    if query:
        # Get SQL query, results, and interpretation
        sql, results, interpretation = sql_prompt.text_to_query(schema, template)

        # Display the SQL query
        with st.expander("View Generated SQL Query", expanded=True):
            st.code(sql, language="sql")

        # Display the results
        st.subheader("Query Results:")
        if not results.empty:
            # Display the interpretation
            st.info(interpretation)

            # Show the results in a table
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Add download button for results
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                csv,
                "query_results.csv",
                "text/csv",
                key="download-csv",
            )
        else:
            st.write("No results found for your query.")

except Exception as e:
    st.error(f"Error: {str(e)}")

# Cleanup temporary file if it was created
if uploaded_db and "tmp_file" in locals():
    try:
        os.unlink(tmp_file.name)
    except:
        pass
