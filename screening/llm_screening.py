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
import sys
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from project_paths import (
    AUDIT_LOG_PATH,
    FILTERED_PAPERS_PATH,
    METADATA_PATH,
    PROCESSED_DIR,
    SCREENING_LOG_PATH,
)
from run_manifest import RunManifest



# ------------------------------ 1. CONFIGURATION ------------------------------ 
# paths, constants, environment set up

FILTERED_OUTPUT_PATH = FILTERED_PAPERS_PATH
SCREENING_LOG_OUTPUT_PATH = SCREENING_LOG_PATH
manifest = RunManifest("llm_screening")

# load previous included papers if exists
if os.path.exists(FILTERED_OUTPUT_PATH):
    with open(FILTERED_OUTPUT_PATH, "r", encoding="utf-8") as f:
        included_papers = json.load(f)
else:
    included_papers = []

print("=== START SCREENING RUN ===")


# ------------------------------ 2. PROMPTS ------------------------------ 
# LLM prompts for screening and verification

SCREENING_PROMPT = """You are an academic assistant performing literature screening for a research project.

The goal of this screening process is to identify literature relevant to
healthcare research, evidence-based practice, and the use of knowledge
or information within professional healthcare contexts.

The system operates as a conservative pre-screening assistant that
applies rule-based inclusion and exclusion criteria before human review.
Its purpose is to narrow the candidate evidence set while preserving
final evaluative authority with the human researcher.

Your task is to decide whether a scientific paper should be INCLUDED
or EXCLUDED based solely on the title and abstract provided.


Inclusion criteria (INCLUDE if MOST apply):

- The paper concerns healthcare, rehabilitation, clinical research,
  public health, or professional healthcare practice.

- The paper relates to healthcare professionals' use, understanding,
  management, or application of knowledge, research evidence,
  digital tools, or information systems in healthcare contexts.

- The paper discusses evidence-based practice, professional education,
  decision-making, digital health technologies, information systems,
  or knowledge-related processes in healthcare.

- Studies involving AI systems, digital platforms, decision-support
  tools, or health technologies are acceptable when they are discussed
  in relation to healthcare professionals, healthcare systems,
  professional practice, or healthcare research contexts.


Exclusion criteria (EXCLUDE if ANY apply):

- The paper is directly addressed to patients or primarily studies
  patient behaviour, engagement, or patient-facing applications.

- The paper describes treatment delivery, therapeutic interventions,
  or clinical procedures performed on patients as actionable care.

- The paper evaluates patient outcomes, treatment effectiveness,
  or clinical efficacy of medical or rehabilitation interventions.

- The paper proposes or evaluates AI systems for diagnosis,
  prediction of clinical outcomes, or autonomous clinical decision-making.

- The paper focuses exclusively on non-healthcare domains
  (e.g., finance, digital currency, general blockchain infrastructure)
  without clear relevance to healthcare practice.
  
- Exclude non-English papers


Important constraints:

- Do NOT summarise the paper.
- Do NOT evaluate scientific quality.
- Do NOT provide recommendations.
- Do NOT add extra commentary.

If there is uncertainty, output EXCLUDE.

Your output must follow this exact format:

Decision: INCLUDE or EXCLUDE
Justification: 1–2 sentences explaining which criteria were applied.


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
manifest.add_event(
    "loaded_metadata",
    METADATA_PATH,
    {"paper_count": len(papers)},
)


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

load_dotenv()

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


# ------------------------------ 6. AUDIT LOGGING ------------------------------ 
# Records screening decisions for traceability

def write_audit_entry(paper_id, decision, validation_status):

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "paperId": paper_id,
        "decision": decision,
        "validation": validation_status
    }

    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
    else:
        audit_data = []

    audit_data.append(entry)

    with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=4)

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

os.makedirs(PROCESSED_DIR, exist_ok=True)
filtered_existed = os.path.exists(FILTERED_OUTPUT_PATH)
screening_log_existed = os.path.exists(SCREENING_LOG_OUTPUT_PATH)
audit_log_existed = os.path.exists(AUDIT_LOG_PATH)
with open(FILTERED_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(included_papers, f, ensure_ascii=False, indent=4)

with open(SCREENING_LOG_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_screening_results, f, ensure_ascii=False, indent=4)

manifest.add_event(
    "updated" if filtered_existed else "created",
    FILTERED_OUTPUT_PATH,
    {"included_count": len(included_papers)},
)
manifest.add_event(
    "updated" if screening_log_existed else "created",
    SCREENING_LOG_OUTPUT_PATH,
    {"screening_entries": len(all_screening_results)},
)
if os.path.exists(AUDIT_LOG_PATH):
    manifest.add_event(
        "updated" if audit_log_existed else "created",
        AUDIT_LOG_PATH,
        {},
    )

print(f"Total INVALID cases: {invalid_cases}")
print("Saved filtered_papers.json and screening_log.json")


print("=== END SCREENING RUN ===")
print(f"Total INCLUDED papers: {len(included_papers)}")


included = sum(1 for r in all_screening_results if r["decision"] == "INCLUDE")
excluded = sum(1 for r in all_screening_results if r["decision"] == "EXCLUDE")
valid = sum(1 for r in all_screening_results if r["validation"] == "VALID")



# print summary statistics for the screening run.
print("\n=== Screening Statistics ===")
print("Total papers:", len(all_screening_results))
print("Included:", included)
print("Excluded:", excluded)
print("Invalid:", invalid_cases)
print("Valid:", valid)


print("\nScreening completed successfully.")
print("Filtered corpus ready for retrieval indexing.")

manifest.set_summary(
    metadata_path=os.path.relpath(METADATA_PATH, BASE_DIR),
    total_papers=len(all_screening_results),
    included=included,
    excluded=excluded,
    invalid=invalid_cases,
    valid=valid,
)
manifest_path = manifest.write()
print(f"Run manifest written to: {manifest_path}")
