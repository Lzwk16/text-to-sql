# Text to SQL Query AI Assistant Converter

This repository consists of a simple application powered by Streamlit that
utilises a locally powered Large Language Model that generates SQL querys from
natural laugange input. It allows you to efficiently create SQl queries to
provide downstream data analysis for business users and stakeholders.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install uv package manager:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Install the required packages and applications:**

    Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Qwen3 model:

    ```bash
    ollama pull qwen3:8b
    ```

    Install required dependencies using uv:

    ```bash
    uv sync
    ```

4.  **Input your raw database files:**

To use your own database table instead of the current file, just
replace it with your desired database table file in the path `db/noshow.db` as
well as its configuration in `config.yaml`


# Run
Run the Streamlit app using uv:

```bash
uv run streamlit run main.py
```