import streamlit as st
from crewAI.rag import build_vectordb
from crewAI.panel import run_panel

st.set_page_config(page_title="IELTS Writing Evaluator", layout="wide")
st.title("📝 IELTS Writing Evaluation (CrewAI + RAG + Confidence)")

essay = st.text_area("Paste your IELTS essay here:", height=300)

if st.button("Evaluate Essay"):
    if not essay.strip():
        st.warning("Please paste an essay first.")
    else:
        with st.spinner("Evaluating..."):
            vectordb = build_vectordb()
            result = run_panel(essay, vectordb)

        # Criterion scores
        st.subheader("📊 Criterion Scores")
        for r in result["criteria"]:
            st.markdown(f"**{r['criterion']}** — Band {r['band']}")
            st.markdown(f"- **Strengths:** {r['strengths']}")
            st.markdown(f"- **Weaknesses:** {r['weaknesses']}")
            st.markdown(f"- **Improvement Tips:** {r['improvement_tips']}")
            st.markdown("---")

        # Chief Examiner
        st.subheader("🎓 Chief Examiner Decision")
        chief = result["chief_examiner"]
        st.markdown(f"- **Final Band:** {chief['final_band']}")
        st.markdown(f"- **Confidence Score:** {chief['confidence_score']}% ({chief['confidence_label']})")
        st.markdown(f"- **Moderation Notes:** {chief['moderation_notes']}")
        if chief["adjustments_made"]:
            st.markdown(f"- **Adjustments Made:** {chief['adjustments_made']}")
