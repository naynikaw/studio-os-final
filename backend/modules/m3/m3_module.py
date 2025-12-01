"""
Module M3: Problem Understanding & Cost Analysis
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
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

def run_pipeline(problem: str) -> str:
    """
    Core pipeline function for M3 module.
    Analyzes the problem and returns JSON string with cost analysis.
    """
    prompt = f"""You are a problem understanding and cost analysis expert.

Task: Analyze the following problem and provide a comprehensive deep dive report.

Problem: {problem}

Provide a detailed analysis in the following JSON format:
{{
  "executive_summary": "A 2-3 sentence summary of the problem and its significance",
  "historical_context": "Background information about how this problem has evolved",
  "root_causes": [
    "Primary root cause 1",
    "Primary root cause 2",
    "Primary root cause 3"
  ],
  "cost_analysis": {{
    "qualitative_costs": [
      "Cost impact 1 (e.g., lost productivity)",
      "Cost impact 2 (e.g., customer churn)",
      "Cost impact 3 (e.g., reputation damage)"
    ],
    "quantitative_estimates": {{
      "annual_cost_range": "Estimated range (e.g., $10M-$50M annually)",
      "methodology": "How the estimate was derived",
      "assumptions": [
        "Assumption 1",
        "Assumption 2"
      ]
    }}
  }},
  "affected_segments": [
    {{
      "segment": "Affected user/business segment",
      "impact": "Description of impact"
    }}
  ],
  "citations": [
    "Source 1 if available",
    "Source 2 if available"
  ],
  "confidence_score": 0.85
}}

Guidelines:
- Be specific and data-driven where possible
- Use realistic cost estimates based on similar problems
- Confidence score should be between 0.0 and 1.0
- If specific data isn't available, note it in assumptions
- Focus on actionable insights

Return ONLY valid JSON, no additional text."""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', result_text, flags=re.DOTALL)
        if json_match:
            return json_match.group(0)
        else:
            # Fallback: return structured JSON even if extraction fails
            return json.dumps({
                "executive_summary": f"Analysis of: {problem}",
                "historical_context": result_text[:500] if result_text else "Analysis in progress",
                "root_causes": [],
                "cost_analysis": {
                    "qualitative_costs": [],
                    "quantitative_estimates": {
                        "annual_cost_range": "To be determined",
                        "methodology": "LLM-based analysis",
                        "assumptions": ["Analysis based on problem description"]
                    }
                },
                "affected_segments": [],
                "citations": [],
                "confidence_score": 0.7
            })
    except Exception as e:
        # Return error structure as JSON string
        return json.dumps({
            "executive_summary": f"Error analyzing problem: {str(e)}",
            "historical_context": "",
            "root_causes": [],
            "cost_analysis": {
                "qualitative_costs": [],
                "quantitative_estimates": {
                    "annual_cost_range": "",
                    "methodology": "",
                    "assumptions": []
                }
            },
            "affected_segments": [],
            "citations": [],
            "confidence_score": 0.0
        })

def run_m3(problem: str) -> dict:
    """
    Main function for M3 module.
    Returns JSON-serializable dict with deep dive report.
    """
    try:
        # Run the pipeline and get JSON string
        json_str = run_pipeline(problem)
        
        # Parse JSON string to dict
        if isinstance(json_str, str):
            return json.loads(json_str)
        else:
            return json_str
    except Exception as e:
        # Return error in structured format
        return {
            "executive_summary": f"Error processing problem: {str(e)}",
            "historical_context": "",
            "root_causes": [],
            "cost_analysis": {
                "qualitative_costs": [],
                "quantitative_estimates": {
                    "annual_cost_range": "",
                    "methodology": "",
                    "assumptions": []
                }
            },
            "affected_segments": [],
            "citations": [],
            "confidence_score": 0.0
        }
