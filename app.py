"""Streamlit client for the text-to-SQL agent API."""

import os
import uuid

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT_S", "90"))

st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.title("Text-to-SQL Agent")

ss = st.session_state
ss.setdefault("db_id", None)
ss.setdefault("db_name", "sample database")
ss.setdefault("thread_id", f"session_{uuid.uuid4().hex[:8]}")
ss.setdefault("history", [])

with st.sidebar:
    st.header("Database")
    st.caption(
        "Privacy: the schema, sample rows and query results are sent to a "
        "cloud LLM (Groq). Do not upload data you cannot share."
    )
    up = st.file_uploader("Upload a SQLite file", type=["db", "sqlite", "sqlite3"])
    if up is not None and st.button("Load database"):
        resp = requests.post(
            f"{API}/upload-db", files={"file": (up.name, up.getvalue())}, timeout=60
        )
        if resp.ok:
            ss.db_id, ss.db_name = resp.json()["db_id"], up.name
            ss.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            ss.history = []
            st.success(f"Loaded {up.name}")
        else:
            st.error(resp.json().get("detail", resp.text))
    if st.button("Use sample database"):
        ss.db_id, ss.db_name, ss.history = None, "sample database", []
    if st.button("New conversation"):
        ss.thread_id, ss.history = f"session_{uuid.uuid4().hex[:8]}", []

    st.header("Options")
    try:
        conv = requests.get(f"{API}/conversions", timeout=30).json()
    except (requests.RequestException, ValueError):
        conv = {"available": [], "default": []}
    labels = {c["name"]: c["label"] for c in conv["available"]}
    ss.conversions = st.multiselect(
        "Convert values stored as text",
        options=list(labels),
        default=[n for n in conv["default"] if n in labels],
        format_func=lambda n: labels.get(n, n),
        help=(
            "For databases that store values as text, such as 'USD$ 12.30', "
            "'5 kg' or '05/10/1973'. Pick 'Auto-detect' to use whatever this "
            "database needs. Each converter only changes columns it recognizes, "
            "and also works inside the SQL (sums, averages, sorting). Your data "
            "is not modified."
        ),
    )

    # Per-converter settings (target units, date order, ...). Shown for the chosen
    # converters; "auto" may enable any of them, so it shows all.
    options_by_name = {c["name"]: c.get("options", []) for c in conv["available"]}
    shown = (
        [n for n in options_by_name if options_by_name[n]]
        if "auto" in ss.conversions
        else [n for n in ss.conversions if options_by_name.get(n)]
    )
    ss.conversion_options = {}
    if shown:
        with st.expander("Conversion settings"):
            for name in shown:
                st.caption(labels.get(name, name))
                for opt in options_by_name[name]:
                    choice = st.selectbox(
                        opt["label"],
                        opt["choices"],
                        index=opt["choices"].index(opt["default"]),
                        key=f"opt_{name}_{opt['key']}",
                    )
                    if choice != opt["default"]:
                        ss.conversion_options.setdefault(name, {})[opt["key"]] = choice

    st.write(f"**Active:** {ss.db_name}")
    try:
        sch = requests.get(f"{API}/schema", params={"db_id": ss.db_id}, timeout=30)
        if sch.ok:
            body = sch.json()
            for t in body["tables"]:
                with st.expander(f"{t['name']} ({t['rows']} rows)"):
                    st.write(", ".join(t["columns"]))
            if body.get("convertible"):
                st.caption(
                    "Convertible columns detected: "
                    + ", ".join(
                        f"{c['table']}.{c['column']} ({c['converter']})"
                        for c in body["convertible"]
                    )
                )
    except requests.RequestException:
        st.error("Backend unreachable. Start it with: uvicorn api:app")


def render(turn: dict) -> None:
    if turn.get("declined"):
        st.info(turn["answer"])
        return
    if turn.get("failed"):
        st.error(turn["answer"])
    else:
        st.markdown(turn["answer"])
    if turn.get("columns"):
        df = pd.DataFrame(turn["rows"], columns=turn["columns"])
        st.dataframe(df, width="stretch")
        if turn.get("truncated"):
            st.caption("Results were capped at the row limit.")
        if turn.get("conversions_applied"):
            for a in turn["conversions_applied"]:
                st.caption(f"Column '{a['column']}' converted {a['detail']} ({a['converter']}).")
        elif turn.get("conversions_requested"):
            st.caption("Conversions are on, but no convertible columns were found.")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "results.csv",
            key=f"dl_{turn['id']}",
        )
    with st.expander(f"Generated SQL and attempts ({turn.get('attempts', 0)})"):
        st.code(turn.get("sql", ""), language="sql")
        for line in turn.get("trace", []):
            st.text(line)


for h in ss.history:
    with st.chat_message("user"):
        st.write(h["question"])
    with st.chat_message("assistant"):
        render(h)

if question := st.chat_input("Ask a question about your data"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API}/chat",
                    json={
                        "message": question,
                        "thread_id": ss.thread_id,
                        "db_id": ss.db_id,
                        "conversions": list(ss.conversions),
                        "conversion_options": ss.conversion_options,
                    },
                    timeout=CHAT_TIMEOUT,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                st.stop()
        if r.ok:
            turn = {
                **r.json(),
                "question": question,
                "id": uuid.uuid4().hex,
                "conversions_requested": bool(ss.conversions),
            }
            ss.history.append(turn)
            render(turn)
        else:
            st.error(r.json().get("detail", r.text))
