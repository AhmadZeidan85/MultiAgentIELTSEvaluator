import json
import re
import yaml
from crewai import Agent, Task, Crew
from crew.llm import call_llm
from crew.rag import retrieve

# ---------- Utilities ----------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group())

AGENTS = load_yaml("config/agents.yaml")
TASKS = load_yaml("config/tasks.yaml")

# ---------- Agents ----------

def examiner_agent(key):
    cfg = AGENTS[key]
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=call_llm,
        verbose=True,
    )

def chief_examiner_agent():
    cfg = AGENTS["chief_examiner"]
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=call_llm,
        verbose=True,
    )

# ---------- Main Pipeline ----------

def run_examiner_panel(essay, vectordb):
    results = {}

    criteria_map = {
        "task_achievement_examiner": "Task Achievement",
        "coherence_cohesion_examiner": "Coherence & Cohesion",
        "lexical_resource_examiner": "Lexical Resource",
        "grammar_examiner": "Grammar",
    }

    # ---- Criterion Examiners ----
    for agent_key, criterion in criteria_map.items():
        rubric_chunks = retrieve(vectordb, criterion)
        rubric_text = "\n".join(c.page_content for c in rubric_chunks)

        agent = examiner_agent(agent_key)

        task = Task(
            description=f"""
Evaluate the IELTS essay for {criterion}.

Rubric:
{rubric_text}

STRICT OUTPUT RULES:
- Return ONLY valid JSON
- No text outside JSON

JSON FORMAT:
{{
  "criterion": "{criterion}",
  "band": number,
  "confidence": number,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "improvement_tips": ["string"],
  "rubric_references": ["string"]
}}
Essay:
{essay}
""",
            expected_output="JSON only",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        raw = crew.kickoff()
        results[criterion] = extract_json(raw)

    # ---- Chief Examiner ----
    chief_agent = chief_examiner_agent()
    examiner_json = json.dumps(results, indent=2)

    chief_task = Task(
        description=f"""
You are the IELTS Chief Examiner.

Examiner reports (JSON):
{examiner_json}

RULES:
- Average bands
- Round to nearest 0.5
- Return ONLY JSON

JSON FORMAT:
{{
  "overall_band": number,
  "confidence": number,
  "final_strengths": ["string"],
  "final_weaknesses": ["string"],
  "top_improvements": ["string"],
  "band_justification": "string"
}}
""",
        expected_output="JSON only",
        agent=chief_agent,
    )

    chief_crew = Crew(
        agents=[chief_agent],
        tasks=[chief_task],
    )

    final_raw = chief_crew.kickoff()
    final_report = extract_json(final_raw)

    return {
        "criteria": results,
        "final_report": final_report,
    }
