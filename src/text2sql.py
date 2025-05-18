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
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [column["name"] for column in columns]

        return json.dumps(schema)

    def text_to_query(self, schema, prompt_template):
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model
        text = chain.invoke({"query": self.query, "schema": schema})
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text.strip()
