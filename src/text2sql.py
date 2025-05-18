import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect


class SQLPromptTemplate:
    def __init__(self, model_name, query):
        self.model = OllamaLLM(model=model_name)
        self.query = query

    def extract_schema(self, db_path):
        """
        Extracts the schema from the given database path.

        The schema is a JSON object, where the keys are the table names,
        and the values are lists of column names.

        Parameters
        ----------
        db_path : str
            The path to the SQLite database file.

        Returns
        -------
        str
            The extracted schema as a JSON object.
        """
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [column["name"] for column in columns]

        return json.dumps(schema)

    def text_to_query(self, schema, prompt_template):
        """
        Converts a natural language query into a SQL query.

        Parameters
        ----------
        schema : str
            The extracted schema of the database as a JSON object.
        prompt_template : str
            The prompt template to use for the model.

        Returns
        -------
        str
            The generated SQL query.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model
        text = chain.invoke({"query": self.query, "schema": schema})
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text.strip()
