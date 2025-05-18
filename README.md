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

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages and applications:**

    Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

    ```bash
    ollama pull deepseek-r1:8b
    ```

    Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Input your raw database files:**

To use your own database table instead of the current file, just
replace it with your desired database table file in the path `db/noshow.db`.


# Run
Run the Streamlit app:

```bash
streamlit run main.py
```