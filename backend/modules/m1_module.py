import os
from openai import OpenAI
from typing import Optional, Dict, Any

# Prompt from constants.ts
PROMPT_MOD_1 = """
Primary Objective: To identify and articulate non-obvious, high-potential problem statements from the noise of public discourse, grounded in credible, emerging trends.

Strategic Mindset: Think like a venture capital scout meeting an investigative journalist. Synthesize disparate signals into a coherent and investable point of view. Prioritize novelty and scale.

Action Required:
Use Google Search to validate trends. Look for:
- Recent Arxiv papers or Google Scholar citations regarding the theme.
- Discussions on HackerNews, Reddit (r/technology, r/startups), or Twitter.
- Industry reports (McKinsey, Gartner, State of AI).

Process:
1. Signal Gathering: Based on the input theme, initiate a multi-vector search.
2. Synthesis: Triangulate findings.
3. Distill: Create clear problem statements framed from the perspective of the entity in pain.
4. Internal Scoring (Hidden): Evaluate Novelty, Scale, Evidence.

Deliverable: A concise brief titled "Emerging Opportunity Analysis." Present the top 3 ranked problem statements. For each, provide a 2-3 sentence paragraph explaining the underlying trend and evidence. CITING SOURCES IS MANDATORY.
"""

def run_m1(problem: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Module 1: Problem Discovery & Trend Detection
    """
    # Resolve API Key
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY environment variable or pass it explicitly.")

    client = OpenAI(api_key=resolved_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    system_instruction = f"""
    You are an expert entrepreneur, product strategist, and venture analyst.
    
    {PROMPT_MOD_1}
    """

    user_prompt = f"Theme/Problem Focus: {problem}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        return {
            "output": content,
            "sources": [] # M1 in this simple version doesn't have structured sources yet, but we return the structure
        }

    except Exception as e:
        return {
            "error": str(e),
            "output": f"Failed to generate analysis: {str(e)}"
        }
