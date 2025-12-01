"""
Module M4: Current Solutions Analysis
Refactored for FastAPI integration
"""
import os
from openai import OpenAI
import json

API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """You are a product strategy and market research expert.

Task: Based on the app description below, perform a real-time competitive analysis using current, trustworthy sources like crunchbase.

App Description:
{APP_DESCRIPTION}

Scope & Requirements:
1) Identify 6–10 relevant competitors (direct + adjacent). Prioritize those serving the specified target users/market on the stated platforms with overlapping value propositions.
2) For each competitor, include:
   - Name + 1-line description
   - Feature overlap with my app (bullet list)
   - Unique differentiators
   - Strengths vs. my app
   - Weaknesses vs. my app
   - Recent updates / market news / trends (with dates)
   - Sources/citations (links)
3) Overall Landscape Summary:
   - Key opportunities for my app
   - Major risks/threats
   - Barriers to entry & switching costs
   - Pricing/monetization patterns observed
   - Regulatory or platform policy considerations (if any)
   - Open whitespace & unmet user needs
4) Strategy:
   - 3 strategic moves (e.g., focus, wedge, GTM)
   - 90-day action plan (milestones + metrics)

Output Format (use these headings exactly):
- Top Competitors (6–10)
  - [Competitor]: Short Description
    - Overlap:
    - Differentiators:
    - Strengths vs. Us:
    - Weaknesses vs. Us:
    - Recent Updates (dated):
    - Sources:
- Landscape Summary
  - Opportunities:
  - Risks/Threats:
  - Barriers & Switching Costs:
  - Pricing Patterns:
  - Regulatory/Policy Notes:
  - Whitespace & Unmet Needs:
- Strategy
  - 3 Strategic Moves:
  - 90-Day Action Plan (with metrics):

Guidelines:
- Browse the web for the latest info; cite sources inline. Prefer primary sources, reputable tech media, company blogs, app store listings.
- Date everything time-sensitive (e.g., "Updated September 2025").
- If evidence is thin, note uncertainty and suggest how to validate.
- Focus on drivers of advantage: acquisition, retention, monetization, defensibility.

My Constraints/Goals:
- Target user & JTBD: {TARGET_USER_JTBD}
- Geography: {GEOGRAPHY}
- Platform(s): {PLATFORMS}
- Business model: {BUSINESS_MODEL}
- Differentiators I'm aiming for: {TARGET_DIFFERENTIATORS}

IMPORTANT: Provide concise, decision-ready analysis with clear citations and dates.
"""

def build_prompt(
    app_description: str,
    target_user_jtbd: str = "College students; stay on top of classes and tasks",
    geography: str = "U.S. + Canada",
    platforms: str = "iOS, Android, Web",
    business_model: str = "Freemium with subscription upsell",
    target_differentiators: str = "AI study planning, campus/LMS integrations, group discovery"
) -> str:
    return PROMPT_TEMPLATE.format(
        APP_DESCRIPTION=app_description.strip(),
        TARGET_USER_JTBD=target_user_jtbd,
        GEOGRAPHY=geography,
        PLATFORMS=platforms,
        BUSINESS_MODEL=business_model,
        TARGET_DIFFERENTIATORS=target_differentiators
    )

def run_m4(problem: str) -> dict:
    """
    Main function for M4 module.
    Returns JSON-serializable dict with competitive analysis.
    """
    try:
        client = OpenAI(api_key=API_KEY)
        
        prompt = build_prompt(
            app_description=problem,
            target_user_jtbd="First-year college students; reduce overwhelm and improve grades",
            geography="United States",
            platforms="iOS and Android",
            business_model="Freemium; $6.99/mo premium",
            target_differentiators="AI-driven study plans, Canvas/Blackboard integration, intelligent group matching"
        )

        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a senior product strategist who cites recent sources with dates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        result_text = response.choices[0].message.content
        
        # Return as structured dict
        return {
            "analysis": result_text,
            "problem": problem,
            "model": "gpt-4o-mini"
        }
    except Exception as e:
        return {
            "error": str(e),
            "analysis": "",
            "problem": problem
        }

