"""
Module M5: Idea Generation
Refactored for FastAPI integration
"""
import os
import json
from typing import Dict, Tuple

from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please configure it in your environment."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

ARTIFACT_MD_BEGIN = "<MARKDOWN>"
ARTIFACT_MD_END = "</MARKDOWN>"
ARTIFACT_JSON_BEGIN = "<JSON>"
ARTIFACT_JSON_END = "</JSON>"


def _prepare_inputs(problem: str) -> Dict[str, str]:
    """
    Normalize inputs coming from the API. Accepts either plain text or JSON.
    """
    default_text = (problem or "").strip()
    prepared = {
        "problem_statement": default_text,
        "problem_deep_dive": default_text,
        "current_solutions": "",
    }

    if not problem:
        return prepared

    try:
        parsed = json.loads(problem)
    except json.JSONDecodeError:
        return prepared

    if isinstance(parsed, dict):
        prepared["problem_statement"] = str(
            parsed.get("prompt")
            or parsed.get("problem")
            or parsed.get("problem_statement")
            or default_text
        ).strip()
        prepared["problem_deep_dive"] = str(
            parsed.get("problem_deep_dive_report")
            or parsed.get("problemDeepDive")
            or parsed.get("m3Report")
            or prepared["problem_statement"]
        ).strip()
        prepared["current_solutions"] = str(
            parsed.get("current_solutions_landscape")
            or parsed.get("currentSolutions")
            or parsed.get("m4Report")
            or ""
        ).strip()

    return prepared


def _generate(prompt: str, label: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Good Futures venture ideation engine. "
                    "Produce thoughtful, well-structured outputs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.75,
    )
    text = response.choices[0].message.content.strip()
    if not text:
        raise ValueError(f"{label} returned an empty response.")
    return text


def _parse_artifact_blocks(artifact_text: str) -> Tuple[str, Dict]:
    markdown_block = ""
    json_block: Dict = {}

    if ARTIFACT_MD_BEGIN in artifact_text and ARTIFACT_MD_END in artifact_text:
        markdown_block = artifact_text.split(ARTIFACT_MD_BEGIN, 1)[1].split(
            ARTIFACT_MD_END, 1
        )[0].strip()

    if ARTIFACT_JSON_BEGIN in artifact_text and ARTIFACT_JSON_END in artifact_text:
        json_str = artifact_text.split(ARTIFACT_JSON_BEGIN, 1)[1].split(
            ARTIFACT_JSON_END, 1
        )[0].strip()
        try:
            json_block = json.loads(json_str)
        except json.JSONDecodeError:
            json_block = {"raw": json_str}

    return markdown_block, json_block


def run_m5(problem: str, api_key: str = None) -> dict:
    """
    Main function for M5 module.
    Returns JSON-serializable dict with venture concepts, prioritization, validation,
    and artifact outputs aligned with the Colab workflow.
    """
    # Use passed API key if available
    global client
    if api_key:
        client = OpenAI(api_key=api_key)
    try:
        inputs = _prepare_inputs(problem)
        good_futures_theses = """
1.  **Workforce Development:** Investing in tools and platforms that upskill, protect, and empower the modern workforce.
2.  **Empowerment:** Backing ventures that give individuals more agency, control, and economic power.
3.  **Distribution:** Supporting technologies that broaden access to essential services and opportunities.
"""

        prompt_m5_1 = f"""
**Objective:** Generate innovative venture concepts based on the provided research.

**Context:**
You are an AI venture ideation engine for a venture studio called "Good Futures".
Your goal is to create high-potential venture concepts that address significant problems.
You must align your ideas with the Good Futures investment theses.

**Good Futures Investment Theses:**
{good_futures_theses}

**Input 1: Problem Deep Dive Report:**
{inputs['problem_deep_dive'] or 'Not provided'}

**Input 2: Current Solutions & Gaps Analysis:**
{inputs['current_solutions'] or 'Not provided'}

**TASK:**
1.  **Brainstorm Raw Ideas:** Generate a list of at least 8 raw, diverse ideas that address the identified gaps. Think about different angles: AI, data, community, workflow tools, etc.
2.  **Refine and Structure:** From the raw ideas, select and refine the best 3-5 concepts. For each concept, provide the following details:
    * **Concept Name:** A catchy, descriptive name.
    * **Value Proposition:** A single, clear sentence explaining the core benefit.
    * **Key Differentiators:** 2-3 bullet points explaining how it's different from existing solutions.
    * **Thesis Alignment:** Which Good Futures thesis it aligns with and a brief explanation why.
    * **Initial Features:** A list of 3-5 core features for an MVP.

**Output Format:**
Provide the 3-5 refined concepts in a clear, well-structured narrative.
"""
        venture_concepts_output = _generate(prompt_m5_1, "M5-1 Ideation")

        prompt_m5_2 = f"""
**Objective:** Score the generated venture concepts and select the top 1-2 to pursue.

**Context:**
You are an AI analyst. You need to objectively evaluate the venture concepts based on key criteria.

**Input: Generated Venture Concepts:**
{venture_concepts_output}

**TASK:**
1.  **Create a Scoring Table:** Score each concept from 1 (Low) to 5 (High) on the following criteria:
    * **Novelty**
    * **USP Strength**
    * **Thesis Alignment**
    * **Feasibility (Initial Gut Check)**
2.  **Provide Rationale:** After the table, write a brief rationale explaining your scoring.
3.  **Down-select:** Clearly state which 1 or 2 concepts you recommend prioritizing and why.
"""
        prioritization_output = _generate(prompt_m5_2, "M5-2 Prioritization")

        prompt_m5_3 = f"""
**Objective:** Act as a critical "Innovation & Thesis Fit Assessor" to validate the top-ranked concept.

**Context:**
Your role is to be skeptical and ensure the selected idea is truly innovative, aligned, and feasible.
You must identify potential weaknesses and suggest improvements.

**Input: Prioritization Report and Selected Concept:**
{prioritization_output}

**TASK:**
1.  **Isolate the Top Concept:** Identify the #1 recommended concept from the input.
2.  **Perform Critical Validation:** Write a validation report answering:
    * Novelty check
    * Alignment check
    * Feasibility check
3.  **Provide a Final Score:** Give a "Validation Score" from 1 (Weak) to 10 (Strong).
4.  **Suggest Refinements:** If the score is below 8, provide one concrete prompt to refine the idea.
"""
        validation_output = _generate(prompt_m5_3, "M5-3 Validation")

        prompt_m5_4 = f"""
**Objective:** Generate a final "Venture Concept Pack" in Markdown and JSON formats for the validated top concept.

**Context:**
This is the final output of the ideation module. It needs to be clean, structured, and ready for use in the next stage (Module 6).

**Inputs:**
Original Concepts:
{venture_concepts_output}

Validation Report:
{validation_output}

**TASK:**
1. Identify the final validated concept.
2. Produce a Markdown summary.
3. Produce a JSON summary with the same information.

Respond using this exact template:
<MARKDOWN>
... markdown content ...
</MARKDOWN>
<JSON>
{{"conceptName": "...", "valueProposition": "..."}}
</JSON>
"""
        artifact_output = _generate(prompt_m5_4, "M5-4 Artifact Generation")
        artifact_markdown, artifact_json = _parse_artifact_blocks(artifact_output)

        return {
            "input": {
                "problemStatement": inputs["problem_statement"],
                "problemDeepDive": inputs["problem_deep_dive"],
                "currentSolutions": inputs["current_solutions"],
            },
            "outputs": {
                "concepts": venture_concepts_output,
                "prioritization": prioritization_output,
                "validation": validation_output,
                "artifact": {
                    "markdown": artifact_markdown,
                    "json": artifact_json,
                    "raw": artifact_output,
                },
                "citations": [],
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "raw_problem_input": problem,
        }
