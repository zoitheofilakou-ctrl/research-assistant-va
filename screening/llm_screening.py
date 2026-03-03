import json
from openai import OpenAI

included_papers = []

print("=== START SCREENING RUN ===")


#  prompt
SCREENING_PROMPT_TEMPLATE = """
You are an academic assistant performing literature screening for a research project.

Your task is to decide whether a scientific paper should be INCLUDED or EXCLUDED
based solely on the title and abstract provided.

Inclusion criteria (INCLUDE if ALL apply):
- The paper supports healthcare professionals in developing knowledge, understanding research evidence,
  or improving professional reasoning and judgement.
- The paper focuses on scientific knowledge, synthesis, organisation, or interpretation of healthcare or
  rehabilitation literature, or on knowledge-based / information-support systems for professionals.
- The paper contributes to professional learning, evidence awareness, or reflective practice.
- Intervention-related papers are acceptable ONLY when discussed analytically or reflectively
  for professional understanding, not as actionable guidance.

Exclusion criteria (EXCLUDE if ANY apply):
- The paper is directly addressed to patients.
- The paper describes treatment delivery, therapeutic intervention, or clinical application
  performed on patients as actionable care.
- The paper uses AI for diagnosis, prediction of clinical outcomes, or autonomous/semi-autonomous
  clinical decision-making.
- The paper evaluates patient outcomes rather than professional knowledge or understanding.
- The paper presents interventions in a way that could be interpreted as clinical guidance.

Important constraints:
- Do NOT summarise the paper.
- Do NOT provide recommendations.
- Do NOT add extra commentary.

If there is any uncertainty, output EXCLUDE.
Output exactly INCLUDE or EXCLUDE in uppercase.

Your output must follow this exact format:

Decision: INCLUDE or EXCLUDE
Justification: 1–2 sentences explaining which criteria were applied.


Title:
{title}

Abstract:
{abstract}
"""

# load json
with open("../data/raw/hybrede_metadata_v2.json", "r", encoding="utf-8") as f:

    papers = json.load(f)


# deduplicate by paperID - ensures each paper is screened exactly once
unique_papers = {}
for paper in papers:
    pid = paper.get("paperId")
    if pid:
        unique_papers[pid] = paper

papers = list(unique_papers.values())
print(f"Loaded {len(papers)} unique papers after deduplication.")


#----Helpers---

# data preparation
# extract title and abstract for LLM
# ensure missing abstracts are handled safely
def extract_screening_text(paper):
    title = paper.get("title","").strip()
    abstract = paper.get("abstract")

    if not abstract or not abstract.strip():
        abstract = "No abstract provided"

    return title, abstract



# Minimal LLM call function(template)
def screen_paper_with_llm(title, abstract, client):
    """
    Sends one paper to the LLM for screening.
    Returns raw LLM response text.
    """

    prompt = SCREENING_PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # example model
        messages=[
            {"role": "system", "content": "You are a strict academic screening assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()



def parse_llm_response(response_text):
    """
    Parses the LLM response and extracts decision and justification.
    Returns (decision, justification).
    """

    decision = None
    justification = None

    lines = response_text.splitlines()

    for line in lines:
        line = line.strip()

        if line.startswith("Decision:"):
            decision = line.replace("Decision:", "").strip()

        elif line.startswith("Justification:"):
            justification = line.replace("Justification:", "").strip()

    # decision normalization
    if decision:
        decision = decision.strip().upper()

    if decision not in {"INCLUDE", "EXCLUDE"}:
        decision = "EXCLUDE"
        justification = "Invalid or missing decision format."
    

    if not justification:
        justification = "No valid justification provided."

    return decision, justification


#create client
client = OpenAI()

print(f"Total papers loaded: {len(papers)}")

all_screening_results = []

# TEST 
for idx, paper in enumerate(papers, start=1):
    title, abstract = extract_screening_text(paper)



    result = screen_paper_with_llm(title, abstract, client)
    decision, justification = parse_llm_response(result)

    print(f"[{idx}] {decision} — {justification}")

    all_screening_results.append({
    "paperId": paper.get("paperId"),
    "title": paper.get("title"),
    "decision": decision,
    "justification": justification
    })

    if decision == "INCLUDE":
        included_papers.append({
            "paperId": paper.get("paperId"),
            "title": paper.get("title"),
            "decision": decision,
            "justification": justification
        })

with open("../data/processed/filtered_papers.json", "w", encoding="utf-8") as f:
    json.dump(included_papers, f, ensure_ascii=False, indent=4)

with open("../data/processed/screening_log.json", "w", encoding="utf-8") as f:
    json.dump(all_screening_results, f, ensure_ascii=False, indent=4)

print("Saved filtered_papers.json and screening_log.json")


print("=== END SCREENING RUN ===")
print(f"Total INCLUDED papers: {len(included_papers)}")



