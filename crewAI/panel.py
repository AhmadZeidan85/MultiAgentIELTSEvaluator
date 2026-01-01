import yaml, json
from crewai import Agent, Task, Crew
from crewAI.rag import retrieve
from crewAI.confidence import calculate_confidence

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

agents_cfg = load_yaml("config/agents.yaml")
tasks_cfg = load_yaml("config/tasks.yaml")

def run_panel(essay, vectordb):
    agents = {
        "task": Agent(**agents_cfg["task_examiner"]),
        "coherence": Agent(**agents_cfg["coherence_examiner"]),
        "lexical": Agent(**agents_cfg["lexical_examiner"]),
        "grammar": Agent(**agents_cfg["grammar_examiner"]),
    }

    tasks = []
    for key, agent in agents.items():
        docs = retrieve(vectordb, essay, key)
        rubric = "\n".join(d.page_content for d in docs)

        tasks.append(Task(
            description=f"""
{tasks_cfg[f"{key}_task"]["description"]}

Rubric:
{rubric}

Essay:
{essay}
""",
            agent=agent,
            expected_output="JSON"
        ))

    crew = Crew(agents=list(agents.values()), tasks=tasks, process="parallel")
    results = [json.loads(r) for r in crew.kickoff()]

    # Chief Examiner
    chief_agent = Agent(**agents_cfg["chief_examiner"])
    chief_task = Task(
        description=f"""
{tasks_cfg['chief_examiner_task']['description']}

Criterion Results:
{json.dumps(results, indent=2)}
""",
        agent=chief_agent,
        expected_output="JSON"
    )

    final = Crew(agents=[chief_agent], tasks=[chief_task]).kickoff()
    chief_output = json.loads(final[0])

    # Compute Confidence
    confidence_pct, confidence_label = calculate_confidence(
        results,
        chief_output["adjustments_made"] is not None
    )
    chief_output["confidence_score"] = confidence_pct
    chief_output["confidence_label"] = confidence_label

    return {
        "criteria": results,
        "chief_examiner": chief_output
    }
