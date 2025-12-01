"""
Module M5: Idea Generation
Refactored for FastAPI integration
"""
import os
import json

# Suppress gRPC/ALTS warnings
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")

import google.generativeai as genai

# Configure API key
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBc_ed2uv2ApfdQq1O8igRYqqqtLRWBuNE")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

def run_m5(problem: str) -> dict:
    """
    Main function for M5 module.
    Returns JSON-serializable dict with venture concepts.
    """
    try:
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

**User Problem:**
{problem}

**Good Futures Investment Theses:**
{good_futures_theses}

**TASK:**
1.  **Brainstorm Raw Ideas:** Generate a list of at least 8 raw, diverse ideas that address the identified gaps. Think about different angles: AI, data, community, workflow tools, etc.
2.  **Refine and Structure:** From the raw ideas, select and refine the best 3-5 concepts. For each concept, provide the following details:
    * **Concept Name:** A catchy, descriptive name.
    * **Value Proposition:** A single, clear sentence explaining the core benefit.
    * **Key Differentiators:** 2-3 bullet points explaining how it's different from existing solutions.
    * **Thesis Alignment:** Which Good Futures thesis it aligns with and a brief explanation why.
    * **Initial Features:** A list of 3-5 core features for an MVP.

**Output Format:**
Provide the 3-5 refined concepts in a clear, well-structured format. Return as JSON with the following structure:
{{
  "concepts": [
    {{
      "conceptName": "...",
      "valueProposition": "...",
      "keyDifferentiators": ["...", "..."],
      "thesisAlignment": "...",
      "initialFeatures": ["...", "..."]
    }}
  ]
}}
"""

        response_m5_1 = model.generate_content(prompt_m5_1)
        concepts_text = response_m5_1.text
        
        # Try to extract JSON from response
        try:
            import re
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', concepts_text, flags=re.DOTALL)
            if json_match:
                concepts_data = json.loads(json_match.group(0))
            else:
                # Fallback: create structured response from text
                concepts_data = {
                    "concepts": [{
                        "conceptName": "Generated Concept",
                        "valueProposition": concepts_text[:200],
                        "keyDifferentiators": [],
                        "thesisAlignment": "Workforce Development",
                        "initialFeatures": []
                    }]
                }
        except:
            concepts_data = {
                "concepts": [{
                    "conceptName": "Generated Concept",
                    "valueProposition": concepts_text[:200],
                    "keyDifferentiators": [],
                    "thesisAlignment": "Workforce Development",
                    "initialFeatures": []
                }]
            }
        
        return {
            "problem": problem,
            "concepts": concepts_data.get("concepts", []),
            "raw_output": concepts_text
        }
    except Exception as e:
        return {
            "error": str(e),
            "problem": problem,
            "concepts": []
        }

