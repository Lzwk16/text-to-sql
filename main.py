import streamlit as st
import yaml

from src.text2sql import SQLPromptTemplate

# Load configuration from YAML file
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_path = config["db_path"]
template = config["prompt_template"]
llm_config = config.get("llm")
rag_config = config.get("rag")
currency_config = config.get("currency")

st.title("Natural Language to SQL Query")
st.write("Ask questions about your data in plain English!")

# Input for the query
query = st.text_area("What would you like to know about the data?")

try:
    if query:
        # Initialize SQL prompt template with all configs
        sql_prompt = SQLPromptTemplate(
            query=query,
            llm_config=llm_config,
            use_rag=True,
            rag_config=rag_config,
            currency_config=currency_config,
        )

        # Initialize RAG components
        sql_prompt.initialize_rag(db_path)

        # Get SQL query and results
        sql, results = sql_prompt.text_to_query(
            prompt_template=template, enable_normalization=True
        )

        # Display the SQL query
        with st.expander("View Generated SQL Query", expanded=True):
            st.code(sql, language="sql")

        # Display the results
        st.subheader("Query Results:")
        if results is not None and not results.empty:
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
