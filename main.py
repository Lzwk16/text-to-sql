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
