"""
LLM-assisted literature screening pipeline.

This script performs automated title–abstract screening of scientific papers
based on predefined inclusion and exclusion criteria.

Pipeline stages:

1. Load metadata corpus (titles + abstracts)
2. Deduplicate papers using paperId
3. Screen each paper using an LLM
4. Validate the decision with a second LLM call
5. Log all screening decisions
6. Store INCLUDED papers to create a filtered research corpus

The resulting filtered corpus is later used for the retrieval stage
of the HybReDe RAG system.

Important note:
This system performs automated pre-screening only.
Final evaluation of the literature remains under human control and
the system does not perform clinical decision-making.
"""

import json
import os
from openai import OpenAI
from datetime import datetime


# ------------------------------ 1. CONFIGURATION ------------------------------ 
# paths, constants, environment set up

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data", "hybrede_metadata_v4.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FILTERED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "filtered_papers.json")
SCREENING_LOG_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "screening_log.json")
AUDIT_LOG_PATH = os.path.join(PROCESSED_DIR, "audit_log.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# load previous included papers if exists
if os.path.exists(FILTERED_OUTPUT_PATH):
    with open(FILTERED_OUTPUT_PATH, "r", encoding="utf-8") as f:
        included_papers = json.load(f)
else:
    included_papers = []

print("=== START SCREENING RUN ===")


# ------------------------------ 2. PROMPTS ------------------------------ 
# LLM prompts for screening and verification

SCREENING_PROMPT = """You are an academic assistant performing STRICT literature pre-screening for a healthcare research project.

This system is a conservative, rule-based filtering tool designed to identify literature relevant to healthcare research processes and the use of knowledge in professional contexts.

Your goal is to EXCLUDE papers that are not clearly and directly relevant.

----------------------
CORE INCLUSION LOGIC
----------------------

A paper should be INCLUDED if MOST of the following conditions are satisfied:

1. The paper is situated within a healthcare or clinical research context

AND

2. The paper addresses at least one of the following:

- use of knowledge, research evidence, or information
- evidence-based practice
- decision-making processes in healthcare professionals
- interpretation, application, or management of knowledge in healthcare

AND

3. The relevance to healthcare research or professional practice is DIRECT 
or reasonably inferred from the context (not purely background)

----------------------
STRICT EXCLUSION RULES
----------------------

EXCLUDE if ANY of the following apply:

- The paper focuses on diseases, treatments, or clinical outcomes without discussing knowledge use
- The paper is purely biomedical, experimental, or laboratory-based
- The paper describes patient behavior, patient tools, or patient-facing applications
- The paper discusses general healthcare topics without connection to research or knowledge processes
- The connection to professional practice or research use is indirect or unclear
- The abstract is missing, vague, or does not provide enough information

----------------------
DECISION RULE
----------------------

✔ INCLUDE only if MOST inclusion conditions are clearly satisfied  
✘ EXCLUDE only if criteria are clearly not met 

If uncertain but potentially relevant → INCLUDE
If clearly irrelevant → EXCLUDE

----------------------
OUTPUT FORMAT
----------------------

Decision: INCLUDE or EXCLUDE  
Justification must explicitly reference one inclusion criterion or one exclusion rule.
Do not use vague reasoning such as "generally relevant" or "somewhat related".
The justification must clearly explain WHY the paper meets or does not meet the criteria.

----------------------
INPUT
----------------------

Title:
{title}

Abstract:
{abstract}
"""

VERIFICATION_PROMPT = """You are validating a literature screening decision.

A previous AI system screened a paper based on predefined
inclusion and exclusion criteria.

Your task is NOT to rescreen the paper.

Your task is only to verify whether the decision is logically
consistent with the justification and the criteria.

If the justification contradicts the criteria → INVALID.
If the justification is insufficient or vague → INVALID.

If the decision is logically supported by the justification,
return VALID.

Output format:

Validation: VALID or INVALID
Reason: one short sentence.


Decision:
{decision}

Justification:
{justification}

Title:
{title}

Abstract:
{abstract}"""

print("Metadata path:", METADATA_PATH)
print("Exists:", os.path.exists(METADATA_PATH))

# ------------------------------ 3. DATA LOADING ------------------------------
# Load metadata and perform deduplication
with open(METADATA_PATH, "r", encoding="utf-8") as f:

    papers = json.load(f)


# Deduplicate papers using paperId.
# Some metadata sources may contain duplicate records.
# This ensures that each paper is screened exactly once.
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
# ρeplace missing abstracts with a placeholder so the LLM can still process the paper
def extract_screening_text(paper):
    title = paper.get("title","").strip()
    abstract = paper.get("abstract")

    if not abstract or not abstract.strip():
        abstract = "No abstract provided"

    return title, abstract

# The screening system uses a two-step LLM process:
#
# 1) Screening step
#    The LLM decides whether the paper should be INCLUDED or EXCLUDED
#    based on predefined criteria.
#
# 2) Verification step
#    A second LLM call checks whether the decision is logically
#    consistent with the justification and the criteria.




# ------------------------------ 4. LLM FUNCTIONS ------------------------------
# Functions that call the LLM and parse its response

def screen_paper_with_llm(title, abstract, client):

    prompt = SCREENING_PROMPT.format(
        title=title,
        abstract=abstract
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict academic screening assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        top_p=1
    )

    return response.choices[0].message.content.strip()



def parse_llm_response(response_text):
    """
    Parses the LLM response and extracts decision and justification.
    Returns (decision, justification).
    """

    # Conservative decision logic:
    # if the LLM response does not follow the expected format
    # we default to EXCLUDE to avoid accidental inclusion.
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


def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


# ------------------------------ 5. VERIFICATION ------------------------------
# Secondary check to validate the screening decision

def verify_screening(decision, justification, title, abstract, client):

    verification_prompt = VERIFICATION_PROMPT.format(
        decision=decision,
        justification=justification,
        title=title,
        abstract=abstract
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a verification assistant."},
            {"role": "user", "content": verification_prompt}
        ],
        temperature=0,
        top_p=1
    )

    return response.choices[0].message.content.strip()
#create client
client = OpenAI()

print(f"Total papers loaded: {len(papers)}")

all_screening_results = []

invalid_cases = 0


# load previous screening log if exists
if os.path.exists(SCREENING_LOG_OUTPUT_PATH):
    with open(SCREENING_LOG_OUTPUT_PATH, "r", encoding="utf-8") as f:
        all_screening_results = json.load(f)

    screened_ids = {p["paperId"] for p in all_screening_results}

else:
    screened_ids = set()

audit_data = load_json_file(AUDIT_LOG_PATH, [])


# ------------------------------ 6. AUDIT LOGGING ------------------------------ 
# Records screening decisions for traceability

def write_audit_entry(paper_id, decision, validation_status):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "paperId": paper_id,
        "decision": decision,
        "validation": validation_status
    }
    audit_data.append(entry)

# ------------------------------  7. MAIN SCREENING LOOP ------------------------------ 
# Iterates through papers and applies screening pipeline
for idx, paper in enumerate(papers, start=1):

    if paper.get("paperId") in screened_ids:
        print(f"[{idx}] already screened — skipping")
        continue


    title, abstract = extract_screening_text(paper)

    try:
        result = screen_paper_with_llm(title, abstract, client)
        decision, justification = parse_llm_response(result)
        validation = verify_screening(decision, justification, title, abstract, client)

    except Exception as e:
        print(f"[{idx}] API error: {e}")

        decision = "EXCLUDE"
        justification = "Screening failed due to API error."
        validation = "Validation: INVALID\nReason: Screening error."




    print(f"[{idx}] {decision} — {justification}")
    print(validation)

    

    if "INVALID" in validation:
        invalid_cases += 1
        validation_status = "INVALID"

        print(f"[{idx}]  INVALID screening detected")
        
    else:
        validation_status = "VALID"
        write_audit_entry(
            paper.get("paperId"),
            decision,
            validation_status
        )

    all_screening_results.append({
        "paperId": paper.get("paperId"),
        "paper_index": idx,
        "title": title,
        "decision": decision,
        "justification": justification,
        "validation": validation_status,
        "validation_raw": validation
    })


    # Only papers that are both INCLUDED and VALIDATED
    # are added to the filtered corpus.
    # This filtered corpus will later be indexed by the
    # retrieval system in the RAG pipeline.
    if decision == "INCLUDE" and validation_status == "VALID":
        included_papers.append({
            "paperId": paper.get("paperId"),
            "title": title,
            "decision": decision,
            "justification": justification
        })

with open(FILTERED_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(included_papers, f, ensure_ascii=False, indent=4)

with open(SCREENING_LOG_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_screening_results, f, ensure_ascii=False, indent=4)

with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, ensure_ascii=False, indent=4)

print(f"Total INVALID cases: {invalid_cases}")
print("Saved filtered_papers.json and screening_log.json")


print("=== END SCREENING RUN ===")
print(f"Total INCLUDED papers: {len(included_papers)}")


included = sum(1 for r in all_screening_results if r["decision"] == "INCLUDE")
excluded = sum(1 for r in all_screening_results if r["decision"] == "EXCLUDE")



# print summary statistics for the screening run.
print("\n=== Screening Statistics ===")
print("Total papers:", len(all_screening_results))
print("Included:", included)
print("Excluded:", excluded)
print("Invalid:", invalid_cases)

print("\nScreening completed successfully.")
print("Filtered corpus ready for retrieval indexing.")


