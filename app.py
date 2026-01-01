import streamlit as st
from crew.panel import run_examiner_panel
from crew.rag import build_vectordb

st.set_page_config(page_title="IELTS Writing Evaluator")

st.title("IELTS Writing Evaluator (AI Examiner Panel)")

essay = st.text_area("Paste your IELTS essay here:")

if st.button("Evaluate Essay"):
    with st.spinner("Evaluating..."):
        vectordb = build_vectordb()
        report = run_examiner_panel(essay, vectordb)

    st.subheader("Final Band Score")
    st.json(report["final_report"])

    st.subheader("Criterion Breakdown")
    for k, v in report["criteria"].items():
        st.markdown(f"### {k}")
        st.json(v)
