#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4 — Current Solutions + Reddit VoC (OpenAI + SerpAPI, two-pass labeling)

Install:
  pip install openai pandas python-dateutil openpyxl requests packaging

What it does:
  • LLM-only problem normalization + inferred landscape (OpenAI)
  • Competitive analysis (OpenAI free-form markdown)
  • Reddit VoC via SerpAPI + Reddit JSON:
      - SerpAPI searches Google for Reddit links related to problem
      - Direct Reddit JSON fetching for posts and comments
      - Focuses on finding SOLUTIONS (not problems)
      - PASS 1: heuristic sentiment + bucket flags
      - PASS 2: OpenAI labeling for top-K most-uncertain rows
      - FRS PVI scoring + concise Markdown summary
  • Saves JSON + Markdown + CSV + Excel workbook
"""

import os
import re
import json
import time
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

import requests
import pandas as pd
from openai import OpenAI

# PRAW removed - using SerpAPI + direct Reddit JSON fetching instead

# =========================
# 🔐 API KEYS — from environment variables
# =========================
# OpenAI - will be initialized when needed
OPENAI_API_KEY = None
MODEL = "gpt-5.1"
client = None

def get_openai_client():
    """Initialize OpenAI client lazily to avoid errors on import."""
    global client, OPENAI_API_KEY
    if client is None:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            # Fallback for development if env var not set
            pass
        MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client

# SerpAPI for Reddit search
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "c02ff4b7a167f3f52916715952e88f8e9f2359a7e915c44f941cace50df7a5d7")
USER_AGENT = "MarketOS-M4-SerpAPI/1.0"

# =========================
# Output files
# =========================
OUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "module4_outputs"))
OUT_JSON_LLM = os.path.join(OUT_DIR, "m4_current_solutions.json")
OUT_MD_LLM   = os.path.join(OUT_DIR, "m4_current_solutions.md")
OUT_XLSX     = os.path.join(OUT_DIR, "m4_outputs.xlsx")
OUT_MD_COMP  = os.path.join(OUT_DIR, "m4_competitive_analysis.md")
OUT_JSON_COMP= os.path.join(OUT_DIR, "m4_competitive_analysis.json")
OUT_REDDIT_CSV = os.path.join(OUT_DIR, "m4_reddit_feedback.csv")

# =========================
# Prompts (LLM-only landscape + Competitive analysis)
# =========================
PROBLEM_NORMALIZE_PROMPT = """
Normalize the following PROBLEM STATEMENT into a compact JSON.

Return JSON with:
{
  "problem_summary": "",
  "scope": {"in_scope": ["", ""], "out_of_scope": ["", ""]},
  "stakeholders": {"primary": ["", ""], "secondary": ["", ""]},
  "root_causes_hypotheses": ["", ""],
  "constraints": ["", ""],
  "evaluation_criteria": ["", ""]
}
PROBLEM STATEMENT:
"""

LANDSCAPE_PROMPT = """
Create a Module 4 Current Solutions Analysis for the PROBLEM STATEMENT below.
Use ONLY internal knowledge (no web). Mark items as "inferred" where appropriate.

Return strict JSON:
{
  "solution_categories": [
    {"category": "", "why_relevant": "", "subtypes": ["", ""],
     "representative_products_or_players": [""],
     "typical_features": ["", ""],
     "pricing_bands": ["Free","Per-seat $","Mid-market $$","Enterprise $$$"],
     "strengths": ["", ""], "weaknesses": ["", ""],
     "common_success_metrics": ["", ""]}
  ],
  "feature_themes": [{"feature": "", "user_value": "", "measurement": ""}],
  "feature_matrix": {
     "columns": ["<category:subtype>", "..."],
     "rows": ["<feature theme>", "..."],
     "coverage": [["High|Medium|Low|None", "..."]]
  },
  "gaps_and_opportunities": {
     "unmet_needs": ["", ""], "failure_modes": ["", ""],
     "wedge_opportunities": ["", ""], "hypotheses_to_test": ["", ""]
  },
  "assumptions_and_limits": ["", ""]
}
PROBLEM STATEMENT:
"""

# === Competitive Analysis Prompt Template (problem-only analysis) ===
PROMPT_TEMPLATE_PROBLEM_ONLY = """You are a product strategy and market research expert.

Task: Based on the PROBLEM STATEMENT below, perform a real-time competitive analysis to identify existing solutions that address this problem. Use current, trustworthy sources like Crunchbase, company blogs, and reputable tech media.

Problem Statement:
{PROBLEM_STATEMENT}

{USER_ROLE_CONTEXT}

CRITICAL: First, analyze what TYPE of problem this is:
- If it's about SKILL GAPS, TRAINING, or EDUCATION → Find training programs, courses, educational platforms, professional development solutions
- If it's about TOOLS, SOFTWARE, or PLATFORMS → Find software products, platforms, technical solutions
- If it's about SERVICES → Find consulting firms, service providers, agencies
- If it's about PROCESSES or METHODOLOGIES → Find frameworks, methodologies, process solutions

Then identify solutions in the APPROPRIATE category that directly address the core problem.

Scope & Requirements:
1) Identify 10–15 relevant existing solutions (direct + adjacent competitors). Prioritize solutions that address the stated problem or aspects of it.
2) For each solution/competitor, include:
   - Name + 1-line description
   - How it addresses the problem (feature overlap with problem needs)
   - Strengths and weaknesses in addressing the problem (combine both - what it does well and where it falls short)
   - Recent updates / market news / trends (with dates)
   - Sources/citations (links)

Output Format (use these headings exactly):
- Organize solutions by APPROACH TYPE or SOLUTION CATEGORY
- For each category, provide:
  - ## [Category Name] (e.g., "Structured Training & Education Programs", "Tools & Platforms", "Services & Consulting", "Organizational Changes")
  - Category Description: [1-2 sentences explaining this approach to solving the problem]
  - Solutions in this category:
    - [Company/Solution Name]: Short Description
    - URL: [Website URL]
    - How it addresses the problem:
      - Strengths and weaknesses in addressing the problem:
    - Recent Updates (dated):
    - Sources:

- Create 3-5 categories that represent different approaches to solving the problem
- Each category should have 2-5 solutions
- Total of 10-15 solutions across all categories

Guidelines:
- Browse the web for the latest info; cite sources inline. Prefer primary sources, reputable tech media, company blogs, app store listings.
- Date everything time-sensitive (e.g., "Updated September 2025").
- If evidence is thin, note uncertainty and suggest how to validate.
- Focus on how well each solution addresses the problem statement, not on comparing to a hypothetical "your app".
- For training/education problems, prioritize educational platforms, courses, professional development programs, executive education, certification programs.
- For tool/platform problems, prioritize software products, platforms, technical solutions.

IMPORTANT: 
- Provide ONLY the competitor list with their details. Do NOT include Landscape Summary or Strategic Insights sections.
- Focus on companies and their solutions only.
- Provide concise, decision-ready analysis with clear citations and dates. Analyze solutions relative to the problem, not relative to any specific product.
"""

# === Competitive Analysis Prompt Template (web-research oriented text output) ===
PROMPT_TEMPLATE = """You are a product strategy and market research expert.

Task: Based on the app description below, perform a real-time competitive analysis using current, trustworthy sources like Crunchbase, company blogs, and reputable tech media.

App Description:
{APP_DESCRIPTION}

Scope & Requirements:
1. Identify 5-8 direct competitors or alternative solutions.
2. For each, provide:
   - Name & Website URL (if known)
   - Core Value Proposition
   - Key Features
   - Pricing Model (if public)
   - Target Audience
3. Analyze 3-5 key market trends relevant to this problem.
4. Identify 2-3 "Blue Ocean" opportunities or gaps.

**CRITICAL INSTRUCTION: CITATIONS**
- You MUST provide in-text citations for every claim you make, e.g., "Competitor X has 5M users [1]" or "The market is shifting to AI [2]".
- At the end of your analysis, you MUST provide a "References" section listing all sources used (Competitor websites, TechCrunch, etc.).
- When listing competitors, ensure you provide their URL.

Return a JSON object with this EXACT structure:
{
  "market_overview": "...",
  "competitors": [
    {
      "name": "...",
      "url": "...",
      "description": "...",
      "strengths": ["..."],
      "weaknesses": ["..."]
    }
  ],
  "trends": ["..."],
  "opportunities": ["..."],
  "citations": [
    {"title": "...", "uri": "...", "source": "Web/Crunchbase"}
  ]
}
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

def build_prompt_problem_only(problem_statement: str, user_role: Optional[str] = None) -> str:
    """Build prompt for competitive analysis using only problem statement."""
    # Map user roles to context descriptions
    role_contexts = {
        "founder": "User Context: The person analyzing this is a Founder/Operator building and scaling products. Focus on actionable insights for product development, market positioning, and competitive differentiation.",
        "investor": "User Context: The person analyzing this is an Investor/Analyst evaluating opportunities. Focus on market size, competitive landscape, investment trends, and market gaps that represent opportunities.",
        "student": "User Context: The person analyzing this is a Student/Researcher learning and exploring. Focus on educational resources, learning opportunities, and comprehensive understanding of the solution space.",
        "studio_partner": "User Context: The person analyzing this is a Studio Partner (GoodFutures internal). Focus on strategic insights, market opportunities, and solutions that align with studio objectives."
    }
    
    user_role_context = role_contexts.get(user_role.lower() if user_role else "", "")
    if user_role_context:
        user_role_context = f"\n{user_role_context}\n"
    
    return PROMPT_TEMPLATE_PROBLEM_ONLY.format(
        PROBLEM_STATEMENT=problem_statement.strip(),
        USER_ROLE_CONTEXT=user_role_context
    )

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

# =========================
# Helper — force strict JSON from LLM-only calls
# =========================
def _strict_json(prompt: str) -> Dict[str, Any]:
    sys_msg = "You are a meticulous market analyst. Return ONLY strict JSON. No prose."
    resp = get_openai_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", MODEL),
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:
        fix = get_openai_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[
                {"role": "system", "content": "Return valid JSON ONLY. No prose."},
                {"role": "user", "content": raw},
            ],
            temperature=0,
        )
        m2 = re.search(r"\{[\s\S]*\}", fix.choices[0].message.content.strip())
        return json.loads(m2.group(0) if m2 else fix.choices[0].message.content)

# =========================
# Feature-matrix normalizer
# =========================
def _normalize_feature_matrix(fm: Dict[str, Any]) -> Dict[str, Any]:
    cols = list(fm.get("columns") or [])
    rows = list(fm.get("rows") or [])
    cov  = list(fm.get("coverage") or [])

    num_cols, num_rows = len(cols), len(rows)
    cov = [r if isinstance(r, list) else [] for r in cov]
    fixed_cov = []
    for r in cov:
        r = list(r[:num_cols]) + ["None"] * max(0, num_cols - len(r))
        fixed_cov.append(r)
    if len(fixed_cov) < num_rows:
        fixed_cov += [["None"] * num_cols for _ in range(num_rows - len(fixed_cov))]
    elif len(fixed_cov) > num_rows:
        fixed_cov = fixed_cov[:num_rows]
    if num_cols == 0 or num_rows == 0:
        return {"columns": [], "rows": [], "coverage": []}
    return {"columns": cols, "rows": rows, "coverage": fixed_cov}

# =========================
# Core LLM-only landscape calls
# =========================
def normalize_problem(problem_statement: str) -> Dict[str, Any]:
    return _strict_json(PROBLEM_NORMALIZE_PROMPT + problem_statement)

def build_landscape(problem_statement: str) -> Dict[str, Any]:
    data = _strict_json(LANDSCAPE_PROMPT + problem_statement)
    if "feature_matrix" in data:
        data["feature_matrix"] = _normalize_feature_matrix(data.get("feature_matrix") or {})
    return data

# =========================
# Competitive analysis (web-research prompt; free-form text/markdown)
# =========================
def run_competitive_analysis(
    problem_statement: str,
    app_description: Optional[str] = None,
    target_user_jtbd: Optional[str] = None,
    geography: Optional[str] = None,
    platforms: Optional[str] = None,
    business_model: Optional[str] = None,
    target_differentiators: Optional[str] = None,
    user_role: Optional[str] = None,
    temperature: float = 0.4
) -> str:
    """
    Run competitive analysis. If only problem_statement is provided, analyzes solutions
    relative to the problem. If app_description and other params provided, compares against your app.
    """
    # If only problem_statement provided, use problem-only analysis
    if not app_description and not target_user_jtbd:
        prompt = build_prompt_problem_only(problem_statement, user_role)
    else:
        # Use original prompt if additional parameters provided
        prompt = build_prompt(
            app_description=app_description or "",
            target_user_jtbd=target_user_jtbd or "",
            geography=geography or "",
            platforms=platforms or "",
            business_model=business_model or "",
            target_differentiators=target_differentiators or ""
        )
    
    resp = get_openai_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", MODEL),
        messages=[
            {"role": "system", "content": "You are a senior product strategist who cites recent sources with dates."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# Optional: naive parser to extract a competitors table from the markdown block
def parse_competitors_from_markdown(md_text: str) -> pd.DataFrame:
    lines = md_text.splitlines()
    rows = []
    row = {}
    def flush(r):
        if r:
            rows.append(r.copy())
    for ln in lines:
        if re.match(r"\s*-\s+Top Competitors", ln, flags=re.I):
            continue
        m2 = re.match(r"\s{2,}-\s+\[(.+?)\]:\s*(.+)$", ln)
        if m2:
            if row: flush(row)
            row = {"competitor": m2.group(1).strip(), "description": m2.group(2).strip(),
                   "overlap": "", "differentiators": "", "strengths_vs_us": "",
                   "weaknesses_vs_us": "", "recent_updates": "", "sources": ""}
            continue
        for key, pat in [
            ("overlap", r"\s{4,}-\s+Overlap:\s*(.*)"),
            ("differentiators", r"\s{4,}-\s+Differentiators:\s*(.*)"),
            ("strengths_vs_us", r"\s{4,}-\s+Strengths vs\. Us:\s*(.*)"),
            ("weaknesses_vs_us", r"\s{4,}-\s+Weaknesses vs\. Us:\s*(.*)"),
            ("recent_updates", r"\s{4,}-\s+Recent Updates \(dated\):\s*(.*)"),
            ("sources", r"\s{4,}-\s+Sources:\s*(.*)"),
        ]:
            m3 = re.match(pat, ln)
            if m3 and row is not None:
                val = m3.group(1).strip()
                row[key] = (row.get(key, "") + " " + val).strip()
    if row:
        flush(row)
    return pd.DataFrame(rows)

def parse_competitive_analysis_structured(md_text: str) -> Dict[str, Any]:
    """
    Parse competitive analysis markdown into structured JSON format.
    Handles both category-based format (new) and flat list format (legacy).
    Returns categories with solutions, plus a flat list for backward compatibility.
    """
    if not md_text or not md_text.strip():
        return {
            "categories": [],
            "competitors": [],
            "total_count": 0,
            "markdown_preview": ""
        }
    
    lines = md_text.splitlines()
    categories = []
    competitors = []  # Flat list for backward compatibility
    
    current_category = None
    current_competitor = None
    current_field = None
    in_category_section = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Match category header: ## Category Name
        category_match = re.match(r"^##\s+(.+)$", line_stripped)
        if category_match:
            # Save previous category and competitor if exists
            if current_competitor and current_category:
                current_category["solutions"].append(current_competitor)
                competitors.append(current_competitor)  # Add to flat list
            elif current_category:
                # Category with no solutions yet
                pass
            
            # Start new category
            category_name = category_match.group(1).strip()
            current_category = {
                "category_name": category_name,
                "category_description": "",
                "solutions": []
            }
            current_competitor = None
            current_field = None
            in_category_section = True
            continue
        
        # Match "Category Description:" line
        if current_category and re.match(r"^Category Description:\s*(.+)$", line_stripped, re.I):
            desc_match = re.match(r"^Category Description:\s*(.+)$", line_stripped, re.I)
            if desc_match:
                current_category["category_description"] = desc_match.group(1).strip()
            continue
        
        # Match "Solutions in this category:" or similar (skip this line)
        if current_category and re.search(r"Solutions?\s+in\s+this\s+category", line_stripped, re.I):
            continue
        
        # Process within a category
        if current_category:
            # Check for field headers FIRST (if we have a current competitor)
            field_processed = False
            if current_competitor:
                # Check for field headers
                field_patterns = [
                    (r"^\s*-\s*\**URL\**:\s*(.*)$", "url", re.I),
                    (r"^\s*-\s*URL:\s*(.*)$", "url", re.I),
                    (r"^\s*-\s*\**How it addresses the problem\**:\s*(.*)$", "how_it_addresses_problem", re.I),
                    (r"^\s*-\s*How it addresses the problem:\s*(.*)$", "how_it_addresses_problem", re.I),
                    (r"^\s*-\s*\**Strengths?\s+(in addressing the problem|vs\. Us)\**:\s*(.*)$", "strengths_and_weaknesses", re.I),
                    (r"^\s*-\s*Strengths?\s+(in addressing the problem|vs\. Us):\s*(.*)$", "strengths_and_weaknesses", re.I),
                    (r"^\s*-\s*\**Weaknesses?\s+(in addressing the problem|vs\. Us)\**:\s*(.*)$", "strengths_and_weaknesses", re.I),
                    (r"^\s*-\s*Weaknesses?\s+(in addressing the problem|vs\. Us):\s*(.*)$", "strengths_and_weaknesses", re.I),
                    (r"^\s*-\s*\**Recent Updates?\s*\(dated\)\**:\s*(.*)$", "recent_updates", re.I),
                    (r"^\s*-\s*Recent Updates?\s*\(dated\):\s*(.*)$", "recent_updates", re.I),
                    (r"^\s*-\s*\**Sources?\**:\s*(.*)$", "sources", re.I),
                    (r"^\s*-\s*Sources?:\s*(.*)$", "sources", re.I),
                ]
                
                for pattern, field_name, flags in field_patterns:
                    match = re.match(pattern, line, flags)
                    if match:
                        current_field = field_name
                        # Extract value (group 2 for patterns with capture in group 1, else group 1)
                        if match.lastindex >= 2:
                            value = match.group(2) if match.group(2) and match.group(2).strip() else (match.group(1) if match.group(1) else "")
                        elif match.lastindex >= 1:
                            value = match.group(1) if match.group(1) else ""
                        else:
                            value = ""
                        
                        # Special handling for strengths_and_weaknesses field - combine strengths and weaknesses
                        if field_name == "strengths_and_weaknesses":
                            # Check if line contains "Weaknesses" (switching from Strengths to Weaknesses)
                            is_weaknesses_line = bool(re.search(r"weaknesses?", line_stripped, re.I))
                            existing_value = current_competitor.get("strengths_and_weaknesses", "")
                            
                            if existing_value and is_weaknesses_line:
                                # We already have strengths, now adding weaknesses - combine with separator
                                current_competitor["strengths_and_weaknesses"] = existing_value + " | Weaknesses: " + (value.strip() if value else "")
                            elif existing_value and not is_weaknesses_line:
                                # Continuing with strengths
                                current_competitor["strengths_and_weaknesses"] = existing_value + " " + (value.strip() if value else "")
                            else:
                                # First time setting this field
                                label = "Strengths: " if not is_weaknesses_line else "Weaknesses: "
                                current_competitor["strengths_and_weaknesses"] = label + (value.strip() if value else "")
                        else:
                            current_competitor[current_field] = value.strip() if value else ""
                        field_processed = True
                        break
                
                if field_processed:
                    continue

                # If no field header found, treat as continuation of current field
                if current_field and line_stripped:
                    # Make sure this isn't a new competitor or field or section
                    is_new_item = (
                        re.match(r"^\s*-\s+\*\*?\[?([^\]:\*\]]+)\]?:?\*\*?\s*[:\-]?\s*(.+)$", line) or  # New competitor
                        re.match(r"^[-#]+\s+", line)  # Section header
                    )
                    if not is_new_item:
                        # Continuation text
                        # For strengths_and_weaknesses, combine with separator if switching between strengths/weaknesses
                        if current_field == "strengths_and_weaknesses":
                            # Check if this line starts a new section (Weaknesses after Strengths)
                            if re.match(r"^\s*Weaknesses?", line_stripped, re.I):
                                # This is a new field, but we combine it
                                if current_competitor[current_field]:
                                    current_competitor[current_field] += " | Weaknesses: "
                                else:
                                    current_competitor[current_field] = "Weaknesses: "
                                # Extract the rest of the line after "Weaknesses:"
                                rest = re.sub(r"^Weaknesses?:\s*", "", line_stripped, flags=re.I)
                                if rest:
                                    current_competitor[current_field] += rest
                            else:
                                # Regular continuation
                                if current_competitor[current_field]:
                                    current_competitor[current_field] += " " + line_stripped
                                else:
                                    current_competitor[current_field] = line_stripped
                        else:
                            # Regular continuation for other fields
                            if current_competitor[current_field]:
                                current_competitor[current_field] += " " + line_stripped
                            else:
                                current_competitor[current_field] = line_stripped
                        continue

            # Match competitor header: - [Company Name]: Description
            competitor_match = re.match(r"^\s*-\s+\*\*?\[?([^\]:\*\]]+)\]?:?\*\*?\s*[:\-]?\s*(.+)$", line)
            if competitor_match:
                # Save previous competitor if exists
                if current_competitor:
                    current_category["solutions"].append(current_competitor)
                    competitors.append(current_competitor)  # Add to flat list
                
                # Start new competitor
                name = competitor_match.group(1).strip()
                description = competitor_match.group(2).strip()
                current_competitor = {
                    "name": name,
                    "url": "",
                    "description": description,
                    "how_it_addresses_problem": "",
                    "strengths_and_weaknesses": "",
                    "recent_updates": "",
                    "sources": ""
                }
                current_field = None
                continue
            
            # Legacy format: Check if we're entering the competitors section (for backward compatibility)
            if re.search(r"Top Competitors", line, re.I):
                in_category_section = True
                continue
            
            # Check if we've left the section
            if re.match(r"^[-#]+\s+(Landscape Summary|Strategy|Summary)", line, re.I):
                if current_competitor and current_category:
                    current_category["solutions"].append(current_competitor)
                    competitors.append(current_competitor)
                if current_category:
                    categories.append(current_category)
                break

    
    # Don't forget the last competitor and category
    if current_competitor:
        if current_category:
            current_category["solutions"].append(current_competitor)
        competitors.append(current_competitor)
    
    if current_category and current_category not in categories:
        categories.append(current_category)
    
    # Clean up fields - normalize whitespace
    for comp in competitors:
        for key in comp:
            if isinstance(comp[key], str):
                comp[key] = re.sub(r'\s+', ' ', comp[key]).strip()
    
    # Clean up category descriptions
    for cat in categories:
        if isinstance(cat.get("category_description"), str):
            cat["category_description"] = re.sub(r'\s+', ' ', cat["category_description"]).strip()
        for sol in cat.get("solutions", []):
            for key in sol:
                if isinstance(sol[key], str):
                    sol[key] = re.sub(r'\s+', ' ', sol[key]).strip()
    
    # Add relevance scores to competitors (will be populated when problem is available)
    # This is a placeholder - scores will be calculated in run_m4 after we have the problem
    
    # If no categories found but competitors exist, create a default category (legacy format)
    if not categories and competitors:
        categories = [{
            "category_name": "Competitive Solutions",
            "category_description": "Existing solutions that address the problem",
            "solutions": competitors
        }]
    
    return {
        "categories": categories,
        "competitors": competitors,  # Flat list for backward compatibility
        "total_count": len(competitors),
        "markdown_preview": md_text[:1000] + "..." if len(md_text) > 1000 else md_text
    }

# =========================
# Markdown export for LLM-only landscape
# =========================
def to_markdown(problem: Dict[str, Any], landscape: Dict[str, Any]) -> str:
    md = []
    md.append("# Module 4 — Current Solutions Analysis\n")
    md.append("## Problem Summary\n" + problem.get("problem_summary", "") + "\n")

    sc = problem.get("scope", {})
    md.append("### Scope")
    md.append(f"- **In-scope:** {', '.join(sc.get('in_scope', []))}")
    md.append(f"- **Out-of-scope:** {', '.join(sc.get('out_of_scope', []))}\n")

    st = problem.get("stakeholders", {})
    md.append("### Stakeholders")
    md.append(f"- **Primary:** {', '.join(st.get('primary', []))}")
    md.append(f"- **Secondary:** {', '.join(st.get('secondary', []))}\n")

    for key, label in [
        ("root_causes_hypotheses", "Root Causes (Hypotheses)"),
        ("constraints", "Constraints"),
        ("evaluation_criteria", "Evaluation Criteria"),
    ]:
        md.append(f"### {label}")
        for it in problem.get(key, []):
            md.append(f"- {it}")
        md.append("")

    md.append("## Current Solutions Landscape (Inferred)")
    for c in landscape.get("solution_categories", []):
        md.append(f"### {c.get('category','')}")
        for k, v in c.items():
            if isinstance(v, list) and v:
                md.append(f"**{k.replace('_',' ').title()}:** " + ", ".join(v))
            elif isinstance(v, str) and v:
                md.append(f"**{k.replace('_',' ').title()}:** {v}")
        md.append("")

    md.append("## Feature Themes")
    for t in landscape.get("feature_themes", []):
        md.append(f"- **{t.get('feature','')}** — value: {t.get('user_value','')} (measure: {t.get('measurement','')})")
    md.append("")

    fm = landscape.get("feature_matrix", {})
    cols, rows, cov = fm.get("columns", []), fm.get("rows", []), fm.get("coverage", [])
    if cols and rows and cov:
        md.append("## Feature Matrix (High / Medium / Low / None)\n")
        md.append("| Feature | " + " | ".join(cols) + " |")
        md.append("|---|" + "|".join(["---"] * len(cols)) + "|")
        for i, r in enumerate(rows):
            vals = cov[i] if i < len(cov) else [""] * len(cols)
            md.append("| " + r + " | " + " | ".join(vals) + " |")
        md.append("")

    gaps = landscape.get("gaps_and_opportunities", {})
    md.append("## Gaps & Opportunities")
    for k, title in [
        ("unmet_needs", "Unmet Needs"),
        ("failure_modes", "Failure Modes"),
        ("wedge_opportunities", "Wedge Opportunities"),
        ("hypotheses_to_test", "Hypotheses to Test"),
    ]:
        if gaps.get(k):
            md.append(f"### {title}")
            for it in gaps[k]:
                md.append(f"- {it}")
            md.append("")

    if landscape.get("assumptions_and_limits"):
        md.append("## Assumptions & Limits")
        for it in landscape["assumptions_and_limits"]:
            md.append(f"- {it}")

    md.append("\n> *LLM-only analysis – no external browsing or citations (model-internal).*")
    return "\n".join(md)

# =========================
# Excel export
# =========================
def to_excel(landscape: Dict[str, Any], comp_markdown: str, reddit_df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Landscape tab
    land_rows = []
    for c in landscape.get("solution_categories", []):
        land_rows.append({
            "category": c.get("category", ""),
            "why_relevant": c.get("why_relevant", ""),
            "subtypes": "; ".join(c.get("subtypes", []) or []),
            "representatives": "; ".join(c.get("representative_products_or_players", []) or []),
            "features": "; ".join(c.get("typical_features", []) or []),
            "pricing": "; ".join(c.get("pricing_bands", []) or []),
            "strengths": "; ".join(c.get("strengths", []) or []),
            "weaknesses": "; ".join(c.get("weaknesses", []) or []),
        })
    df_land = pd.DataFrame(land_rows)

    fm_raw = landscape.get("feature_matrix", {}) or {}
    fm = _normalize_feature_matrix(fm_raw)
    cols, rows, cov = fm.get("columns", []), fm.get("rows", []), fm.get("coverage", [])
    if cols and cov:
        df_fm = pd.DataFrame(cov, columns=cols)
        if rows:
            df_fm.insert(0, "feature_theme", rows)
    else:
        df_fm = pd.DataFrame(columns=["feature_theme"])

    # Competitors parsed (best-effort)
    df_comp = parse_competitors_from_markdown(comp_markdown)

    with pd.ExcelWriter(OUT_XLSX) as xw:
        df_land.to_excel(xw, index=False, sheet_name="Landscape")
        df_fm.to_excel(xw, index=False, sheet_name="Feature Matrix")
        if not df_comp.empty:
            df_comp.to_excel(xw, index=False, sheet_name="Competitors (parsed)")
        else:
            pd.DataFrame({"note": ["Could not parse competitors section; see MD file"]}).to_excel(
                xw, index=False, sheet_name="Competitors (parsed)"
            )
        if reddit_df is not None and not reddit_df.empty:
            reddit_df.to_excel(xw, index=False, sheet_name="Reddit VoC")

# =========================
# ---------- Reddit + OpenAI labeling ----------
# =========================

# Defaults for subreddits and scoring
DEFAULT_SUBS = [
    # hiring & careers
    "recruitinghell", "jobs", "career_advice", "cscareerquestions", "humanresources",
    # AI/ML/data
    "ArtificialIntelligence", "MachineLearning", "DataScience", "ChatGPT", "OpenAI",
    # startups/ops
    "Entrepreneur", "startups", "SaaS", "SideProject", "smallbusiness",
    # work/productivity
    "Productivity", "remotework", "workreform", "GetDisciplined",
    # broad tech
    "technology", "Futurology", "learnprogramming",
    # general voice
    "AskReddit", "explainlikeimfive"
]
F_CAP_DEFAULT = 50
RECENCY_HALF_LIFE = 30.0
WEIGHTS = (0.40, 0.35, 0.25)  # F, R, S
TOTAL_HARD_CAP = 200  # Increased from 100

GENERIC_QUESTION_TRIGGERS = (
    " how ", "how do ", "how can ", " any tips", " tips ", "recommend", "recommendation",
    "suggest", "suggestion", "what should ", "best way", "advice", "any idea", "help?",
    " how?", " where ", " where?", " which ", " which?", " problem?", " fix?"
)
GENERIC_WORKAROUND_TRIGGERS = (
    "i used", "i tried", "i fixed", "i solved", "what worked for me", "i do this", "my workaround",
    "step by step", "steps i took", "script", "macro", "automation", "template", "checklist", "guide"
)
# Removed GENERIC_PAIN_TRIGGERS - only focusing on solutions now

SENT_NEG = {
    "hate","annoy","angry","upset","frustrat","sucks","terrible","awful","bad","pain","hurts",
    "broken","bug","error","issue","problem","fail","failed","worse","hard","stuck","blocked",
    "confusing","confused","overwhelmed","anxious","worried","panic","stress","stressed","rant"
}
SENT_POS = {
    "love","amazing","great","awesome","good","works","fixed","helped","improved",
    "clear","better","success","win","happy","nice","recommend","useful","effective"
}

def utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def days_ago(ts: float) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days

def clamp(v, a, b): return max(a, min(b, v))
def short(s, n=220):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n-1] + "…"

# =========================
# SerpAPI + Reddit JSON fetching functions
# =========================

def serpapi_search(query: str, max_results: int = 10) -> List[str]:
    """Search Google via SerpAPI for Reddit links related to the query."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY is required. Please set it as an environment variable.")
    
    # Validate API key format (SerpAPI keys are typically 64 characters)
    if len(SERPAPI_KEY) < 40:
        logging.warning(f"[serpapi] API key appears invalid (length: {len(SERPAPI_KEY)})")
    
    params = {
        "engine": "google",
        "q": query,
        "num": 10,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30,
                        headers={"User-Agent": USER_AGENT})
        
        # Check for HTTP errors
        if r.status_code != 200:
            error_msg = f"HTTP {r.status_code}"
            try:
                error_data = r.json()
                if "error" in error_data:
                    error_msg = error_data.get("error", error_msg)
                logging.error(f"[serpapi] HTTP error {r.status_code} for '{query}': {error_msg}")
            except:
                logging.error(f"[serpapi] HTTP error {r.status_code} for '{query}': {r.text[:200]}")
            return []
        
        data = r.json()
        
        # Check for SerpAPI-specific errors in response
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            # Check for common SerpAPI error types
            if "Invalid API key" in error_msg or "authentication" in error_msg.lower():
                logging.error(f"[serpapi] Authentication error for '{query}': {error_msg}. Check SERPAPI_KEY.")
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                logging.error(f"[serpapi] Rate limit/quota exceeded for '{query}': {error_msg}")
            else:
                logging.error(f"[serpapi] API error for '{query}': {error_msg}")
            return []
        
        # Check if we have organic_results
        if "organic_results" not in data:
            logging.warning(f"[serpapi] No 'organic_results' in response for '{query}'. Response keys: {list(data.keys())[:5]}")
            return []
        
        urls: List[str] = []
        organic_results = data.get("organic_results", [])
        
        if not organic_results:
            logging.info(f"[serpapi] No organic results found for '{query}'")
            return []
        
        for item in organic_results:
            u = item.get("link")
            if u and re.search(r"(reddit\.com|redd\.it)", u):
                urls.append(u)
                if len(urls) >= max_results:
                    break
        
        if not urls:
            logging.info(f"[serpapi] Found {len(organic_results)} results for '{query}', but none were Reddit links")
        
        return urls
        
    except requests.exceptions.Timeout:
        logging.error(f"[serpapi] Request timeout for '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"[serpapi] Request exception for '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"[serpapi] JSON decode error for '{query}': {e}. Response: {r.text[:200] if 'r' in locals() else 'N/A'}")
        return []
    except Exception as e:
        logging.error(f"[serpapi] Unexpected error for '{query}': {e}", exc_info=True)
        return []

def normalize_reddit_url(url: str) -> str:
    """Normalize Reddit URL to standard format."""
    if re.match(r"https?://(www\.)?redd\.it/[A-Za-z0-9]+/?", url):
        return url.rstrip("/")
    m = re.match(r"https?://(www\.|old\.|np\.)?reddit\.com(/r/[^/]+)?/comments/[a-z0-9]{6,8}/[^/?#]*/?", url)
    if m:
        return m.group(0).rstrip("/")
    m2 = re.match(r"https?://(www\.|old\.|np\.)?reddit\.com/comments/[a-z0-9]{6,8}/?", url)
    if m2:
        return m2.group(0).rstrip("/")
    return url.rstrip("/")

def fetch_reddit_json(url: str, max_attempts: int = 3) -> Optional[List[Any]]:
    """Fetch Reddit post and comments as JSON."""
    jurl = url if url.endswith(".json") else url + ".json"
    h = {"User-Agent": USER_AGENT}
    
    for attempt in range(max_attempts):
        try:
            r = requests.get(jurl, timeout=25, headers=h)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if not r.ok:
                return None
            return r.json()
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                logging.warning(f"[reddit json] Failed to fetch {url}: {e}")
    return None

def extract_post(payload: List[Any]) -> Optional[Dict[str, Any]]:
    """Extract post data from Reddit JSON payload."""
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    
    try:
        sd = payload[0]["data"]["children"][0]["data"]
    except Exception:
        return None
    
    title = (sd.get("title") or "").strip()
    text = (sd.get("selftext") or "").strip()
    hay = f"{title}\n{text}".lower()
    
    # Light promo noise filter
    banned_phrases = [
        "newsletter", "recap", "roundup", "follow me", "sponsor", "affiliate",
        "discount", "coupon", "waitlist", "dm me", "promo code", "hiring", "job opening"
    ]
    if any(p in hay for p in banned_phrases):
        return None
    
    # Quality gates for POST
    MIN_POST_WORDS = 20
    MIN_POST_SCORE = 1
    
    if len((title + " " + text).split()) < MIN_POST_WORDS:
        return None
    if int(sd.get("score", 0)) < MIN_POST_SCORE:
        return None
    
    # ASCII ratio check
    def ascii_alpha_ratio(s: str) -> float:
        if not s:
            return 1.0
        letters = sum(ch.isalpha() and ord(ch) < 128 for ch in s)
        total = max(1, sum(ch != " " for ch in s))
        return letters / total
    
    if ascii_alpha_ratio(title + " " + text) < 0.6:
        return None
    
    return {
        "kind": "post",
        "subreddit": sd.get("subreddit"),
        "title": title,
        "text": text,
        "author": sd.get("author") or "[deleted]",
        "num_comments": int(sd.get("num_comments", 0)),
        "score": int(sd.get("score", 0)),
        "permalink": f"https://www.reddit.com{sd.get('permalink')}" if sd.get("permalink") else None,
        "created_utc": float(sd.get("created_utc", 0.0))
    }

def flatten_comments(children, depth=0):
    """Flatten nested Reddit comments into a list."""
    out = []
    for c in children or []:
        if c.get("kind") != "t1":
            continue
        d = c.get("data", {})
        out.append({
            "kind": "comment",
            "author": d.get("author") or "[deleted]",
            "body": (d.get("body") or "").strip(),
            "score": int(d.get("score") or 0),
            "created_utc": float(d.get("created_utc") or 0.0),
            "permalink": f"https://www.reddit.com{d.get('permalink')}" if d.get("permalink") else None,
            "depth": depth
        })
        replies = d.get("replies")
        if isinstance(replies, dict):
            kids = replies.get("data", {}).get("children", [])
            out.extend(flatten_comments(kids, depth+1))
    return out

def _terms_from(text: str):
    """Extract meaningful terms, filtering out common stop words."""
    # Common stop words that don't help with relevance
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "up", "about", "into", "through", "during", "including", "against", "among",
        "throughout", "despite", "towards", "upon", "concerning", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "up", "about", "into", "through", "during",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "could", "may", "might", "must", "can", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "can", "will", "just", "should", "now"
    }
    raw = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    # Filter out stop words and very short terms
    meaningful = [t for t in raw if len(t) > 2 and t not in stop_words]
    return meaningful

def build_queries(problem: str, mode: str) -> List[str]:
    """Build SerpAPI queries focused on finding solutions on Reddit."""
    base = (problem or "").strip()
    # Focus on solution-sharing keywords with site:reddit.com
    seeds = [
        f'{base} solution site:reddit.com',
        f'{base} workaround site:reddit.com',
        f'how to solve {base} site:reddit.com',
        f'how to fix {base} site:reddit.com',
        f'{base} solved site:reddit.com',
        f'what worked for {base} site:reddit.com',
        f'{base} guide site:reddit.com',
        f'{base} tutorial site:reddit.com',
        f'best way to {base} site:reddit.com',
        f'{base} recommendations site:reddit.com',
        f'{base} tips site:reddit.com',
    ]
    if mode == "fast":
        seeds = seeds[:6]  # Keep top solution-focused queries
    out, seen = [], set()
    for q in seeds:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def heuristic_sentiment(title: str, text: str) -> float:
    def _tok(s: str): return re.findall(r"[a-z0-9]+", (s or "").lower())
    toks = _tok(title) + _tok(text)
    pos = sum(any(p in t for p in SENT_POS) for t in toks)
    neg = sum(any(n in t for n in SENT_NEG) for t in toks) * 1.2
    denom = pos + neg + 5.0
    score = (pos - neg) / denom
    return float(max(-1.0, min(1.0, score)))

def scrape_reddit(problem: str,
                  subreddits: List[str] = None,
                  time_filter: str = "month",
                  mode: str = "fast",
                  per_query_limit: int = 15,
                  max_per_sub: int = 50) -> List[Dict]:
    """
    Scrape Reddit posts using SerpAPI to find Reddit links, then fetch JSON directly.
    Focuses on finding SOLUTIONS to the problem statement.
    """
    queries = build_queries(problem, mode)
    base_terms = set(_terms_from(problem))

    rows, seen_links = [], set()
    kept, skipped_promo, skipped_irrelevant = 0, 0, 0

    # Collect Reddit URLs from SerpAPI searches
    all_urls = []
    queries_attempted = 0
    queries_successful = 0
    
    logging.info(f"[serpapi] Starting search with {len(queries)} queries for problem: '{problem[:100]}'")
    
    for q in queries:
        if len(all_urls) >= TOTAL_HARD_CAP:
            break
        queries_attempted += 1
        urls = serpapi_search(q, max_results=per_query_limit)
        if urls:
            queries_successful += 1
            logging.info(f"[serpapi] Query '{q[:60]}...' returned {len(urls)} Reddit URLs")
        else:
            logging.warning(f"[serpapi] Query '{q[:60]}...' returned no Reddit URLs")
        
        for url in urls:
            normalized = normalize_reddit_url(url)
            if normalized not in seen_links:
                seen_links.add(normalized)
                all_urls.append(normalized)
                if len(all_urls) >= TOTAL_HARD_CAP:
                    break
        time.sleep(0.3)  # Rate limiting for SerpAPI
    
    logging.info(f"[serpapi] Completed: {queries_successful}/{queries_attempted} queries successful, {len(all_urls)} unique Reddit URLs found")
    
    # Deduplicate by thread ID
    dedup_urls = []
    seen_threads = set()
    for url in all_urls:
        key_m = re.search(r"/comments/([a-z0-9]{6,8})/", url)
        key = key_m.group(1) if key_m else url
        if key not in seen_threads:
            seen_threads.add(key)
            dedup_urls.append(url)
            if len(dedup_urls) >= TOTAL_HARD_CAP:
                break
    
    # Fetch posts from Reddit JSON
    for url in dedup_urls:
        if len(rows) >= TOTAL_HARD_CAP:
            break
        
        payload = fetch_reddit_json(url)
        if not payload:
            time.sleep(0.2)
            continue
        
        post = extract_post(payload)
        if not post:
            time.sleep(0.2)
            continue
        
        title = post["title"]
        text = post["text"]
        hay = (title + " " + text).lower()

        # Relevance gate - stricter matching for solutions
        if base_terms:
            problem_lower = problem.lower()
            
            # First check: full problem phrase match (highest priority)
            if problem_lower in hay:
                taken = True
            else:
                # Count how many meaningful terms from problem appear in post
                hits = sum(1 for t in base_terms if t in hay)
                total_terms = len(base_terms)
                
                # For short problems (1-2 terms), require all terms
                if total_terms <= 2:
                    taken = all(t in hay for t in base_terms)
                else:
                    # For longer problems, require majority of terms (at least 60%)
                    min_required = max(2, int(total_terms * 0.6))
                    taken = (hits >= min_required)
        else:
            taken = True
        if not taken:
            skipped_irrelevant += 1
            time.sleep(0.2)
            continue

        # Check if post contains solution (quick check before adding)
        post_has_solution = is_solution_content(text, title)

        rows.append({
            "source": "reddit",
            "subreddit": post["subreddit"],
            "query": "",  # Not applicable for SerpAPI
            "title": title,
            "text": text,
            "author": post["author"],
            "num_comments": post["num_comments"],
            "score": post["score"],
            "permalink": post["permalink"],
            "created_utc": post["created_utc"],
            "has_solution_in_post": post_has_solution,
        })
        kept += 1
        time.sleep(0.2)
    
    logging.info(f"[serpapi reddit] kept={kept}  skipped_irrelevant={skipped_irrelevant}")
    return rows[:TOTAL_HARD_CAP]

# ---------- OpenAI labeling (replaces Gemini) ----------
CLS_PROMPT = """Classify a short social post about a problem theme.
Return STRICT JSON (one object):
{
  "category": "Workaround" | "Solution Seeking",
  "intent": "workaround" | "ask",
  "sentiment_compound": float in [-1.0, 1.0],
  "emotional_intensity": integer 1..5,
  "short_quote": string
}
Guidance:
- Workaround: user describes steps they took that others can reuse (actual solution).
- Solution Seeking: user asks for help, advice, or recommendations (question, might have solution comments).
Only classify as Workaround if it contains an actual actionable solution with steps/methods.
Return JSON ONLY.
"""

def _probable_intent_from_text(title: str, text: str, qtrigs, wtrigs) -> str:
    t = f"{title}\n{text}".lower()
    if any(k in t for k in wtrigs): return "workaround"
    if "?" in t or any(k in t for k in qtrigs): return "ask"
    return ""

def _infer_flags(title: str, text: str, sentiment_compound: float,
                 qtrigs, wtrigs) -> Tuple[bool,bool]:
    """Infer if text is solution-seeking or contains workaround. No pain detection."""
    t = f"{title}\n{text}".lower()
    is_seek = ("?" in t) or any(k in t for k in qtrigs)
    is_work = any(k in t for k in wtrigs) or is_solution_content(text, title)
    return is_seek, is_work

def openai_label(problem: str, title: str, text: str, subreddit: str,
                 qtrigs, wtrigs) -> Dict:
    hint = _probable_intent_from_text(title, text, qtrigs, wtrigs)
    content = (
        f"Problem theme: {problem}\n"
        f"Subreddit: r/{subreddit}\n"
        f"Title: {title}\n"
        f"Body:\n{text}\n"
        f"(Heuristic hint: {hint or 'none'})"
    )
    try:
        resp = get_openai_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[
                {"role": "system", "content": "You classify social posts and return STRICT JSON only."},
                {"role": "user", "content": CLS_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.2
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw = ""

    m = re.search(r"\{.*\}", raw, flags=re.S)
    out = {
        "category": "Solution Seeking", "intent": "ask",
        "sentiment_compound": 0.0, "emotional_intensity": 2,
        "short_quote": short(text or title)
    }
    if m:
        try:
            obj = json.loads(m.group(0))
            cat = obj.get("category", "Solution Seeking")
            if cat not in ("Solution Seeking","Workaround"): 
                cat = "Workaround" if is_solution_content(text, title) else "Solution Seeking"
            intent = obj.get("intent", "ask")
            if cat == "Workaround":
                intent = "workaround"
            sent = float(obj.get("sentiment_compound", 0.0))
            intensity = max(1, min(5, int(obj.get("emotional_intensity", 2))))
            quote = short(obj.get("short_quote") or text or title)
            out = {"category": cat, "intent": intent,
                   "sentiment_compound": sent, "emotional_intensity": intensity,
                   "short_quote": quote}
        except Exception:
            pass

    # Post-fix: if hint suggests workaround but classified as seeking, check if it's actually a solution
    if out["category"] == "Solution Seeking" and hint == "workaround":
        if is_solution_content(text, title):
            out["category"] = "Workaround"; out["intent"] = "workaround"
    return out

def is_solution_content(text: str, title: str = "") -> bool:
    """
    Check if text contains an actual actionable solution/workaround.
    Returns True if text describes HOW to solve something with specific steps/actions.
    Made less strict to capture more valid solutions.
    """
    if not text or len(text.strip()) < 30:  # Reduced from 50 to 30
        return False
    
    combined = f"{title} {text}".lower()
    
    # Exclude complaints/pain - if it's mostly complaining, it's not a solution
    complaint_indicators = [
        "hate", "annoy", "angry", "upset", "frustrat", "sucks", "terrible", "awful",
        "broken", "failed", "doesn't work", "didn't work", "problem is", "issue is",
        "struggling", "confused", "overwhelmed", "difficult", "hard", "pain"
    ]
    complaint_count = sum(1 for indicator in complaint_indicators if indicator in combined)
    # Only exclude if it's heavily complaint-focused (increased threshold)
    if complaint_count >= 5:  # Increased from 3 to 5
        return False
    
    # Solution indicators - expanded to catch more patterns
    solution_patterns = [
        # Strong patterns (required for high confidence)
        r'(i\s+(used|tried|fixed|solved|did|implemented|applied|followed|took|recommend|suggest))',
        r'(what\s+worked\s+(for\s+me|in\s+my\s+case))',
        r'(here\'?s\s+(how|what)\s+(i|to|you))',
        r'(step\s+\d+|step\s+by\s+step|first\s+.*then|next\s+.*finally)',
        r'(try\s+(this|these|using|doing|it))',
        r'(you\s+can\s+(try|use|do|implement|apply|follow|check))',
        r'(solution\s+(is|was|to|involves))',
        r'(workaround\s+(is|was|to|involves))',
        r'(the\s+way\s+(to|i|you))',
        r'(method\s+(that|i|to))',
        r'(approach\s+(that|i|to))',
        # Additional patterns for recommendations and suggestions
        r'(give\s+\w+\s+a\s+try)',
        r'(i\s+(use|used|recommend|suggest|love|prefer))',
        r'(check\s+out\s+\w+)',
        r'(look\s+into\s+\w+)',
        r'(\w+\s+is\s+(great|good|helpful|useful|effective|works))',
    ]
    
    # Must match at least one solution pattern
    has_solution = any(re.search(pattern, combined, re.I) for pattern in solution_patterns)
    
    if not has_solution:
        return False
    
    # Must have actionable content (verbs or recommendations)
    action_indicators = [
        r'\b(do|try|use|install|download|apply|follow|implement|set\s+up|configure|create|build|make|write|enable|disable|add|remove|change|update|modify|check|look|give|get|find|install|download)\b',
        r'\b(recommend|suggest|prefer|love|works|helpful|useful|effective)\b',
    ]
    has_action = any(re.search(pattern, combined, re.I) for pattern in action_indicators)
    
    # Must be substantial (reduced requirement)
    is_substantial = len(text.split()) >= 20  # Reduced from 30 to 20
    
    # Must NOT be just asking a question (relaxed)
    is_question = (combined.count("?") >= 3) or any(q in combined for q in ["how do i", "what should i", "can anyone help", "does anyone know"])
    
    # Return True if it has solution pattern, action, and is substantial, and not just a question
    return has_solution and has_action and is_substantial and not is_question

def _is_relevant_to_problem(text: str, problem: str, base_terms: set) -> bool:
    """
    Check if text is relevant to the problem statement.
    Returns True if the text contains the problem phrase or enough meaningful terms.
    """
    if not text or not problem:
        return False
    
    text_lower = text.lower()
    problem_lower = problem.lower()
    
    # First check: full problem phrase match (highest priority)
    if problem_lower in text_lower:
        return True
    
    # Second check: meaningful terms overlap (relaxed threshold)
    if base_terms:
        hits = sum(1 for t in base_terms if t in text_lower)
        total_terms = len(base_terms)
        
        # For short problems (1-2 terms), require all terms
        if total_terms <= 2:
            return all(t in text_lower for t in base_terms)
        else:
            # For longer problems, require at least 40% of terms (reduced from 60%)
            # This is more lenient to capture more relevant solutions
            min_required = max(1, int(total_terms * 0.4))  # Reduced from 0.6 to 0.4, min 1 instead of 2
            return (hits >= min_required)
    
    return False

def extract_comments_with_solutions(post_url: str, problem: str, max_comments: int = 20) -> List[Dict]:
    """
    Extract comments from a Reddit post that contain solutions RELEVANT to the problem.
    Uses Reddit JSON API instead of PRAW.
    Returns list of comment dicts with solution content.
    """
    solution_comments = []
    
    # Extract meaningful terms from problem for relevance checking
    base_terms = set(_terms_from(problem))
    
    try:
        # Fetch Reddit JSON
        payload = fetch_reddit_json(post_url)
        if not payload or not isinstance(payload, list) or len(payload) < 2:
            return []
        
        # Extract post data to check relevance
        try:
            post_data = payload[0]["data"]["children"][0]["data"]
            post_title = (post_data.get("title") or "").strip()
            post_text = (post_data.get("selftext") or "").strip()
        except Exception:
            return []
        
        post_content = f"{post_title} {post_text}".lower()
        
        # FIRST: Check if the parent post itself is relevant to the problem
        if not _is_relevant_to_problem(post_content, problem, base_terms):
            logging.info(f"[comments] Skipping post '{post_title[:50]}' - not relevant to problem")
            return []
        
        # Extract comments from JSON
        try:
            comments_listing = payload[1]
            children = comments_listing.get("data", {}).get("children", [])
            flat_comments = flatten_comments(children, depth=0)
        except Exception:
            return []
        
        # Filter comments for solutions
        MIN_COMMENT_WORDS = 10
        MIN_COMMENT_SCORE = 1
        
        def ascii_alpha_ratio(s: str) -> float:
            if not s:
                return 1.0
            letters = sum(ch.isalpha() and ord(ch) < 128 for ch in s)
            total = max(1, sum(ch != " " for ch in s))
            return letters / total
        
        def jaccard_3gram(a: str, b: str) -> float:
            """Simple 3-gram Jaccard for near-dup detection."""
            def _ngrams(text: str, n=3):
                toks = re.findall(r"[a-z0-9]+", (text or "").lower())
                return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set()
            A, B = _ngrams(a, 3), _ngrams(b, 3)
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)
        
        comment_count = 0
        dedup_buf = []
        
        for c in flat_comments:
            if comment_count >= max_comments:
                break
                
            body = c.get("body", "").strip()
            if not body or body in ["[deleted]", "[removed]"]:
                continue
            
            if len(body.split()) < MIN_COMMENT_WORDS:
                continue
            
            if c.get("score", 0) < MIN_COMMENT_SCORE:
                continue
            
            if ascii_alpha_ratio(body) < 0.6:
                continue
            
            # Near-duplicate check
            if any(jaccard_3gram(body, prev) > 0.90 for prev in dedup_buf):
                continue
            
            # Check if comment contains a solution
            if not is_solution_content(body):
                continue
            
            # CRITICAL: Check if comment is relevant to the problem
            comment_with_context = f"{post_title} {body}".lower()
            if not _is_relevant_to_problem(comment_with_context, problem, base_terms):
                continue
            
                solution_comments.append({
                "comment_text": body,
                "comment_author": c.get("author", "[deleted]"),
                "comment_score": c.get("score", 0),
                "comment_created_utc": c.get("created_utc", 0.0),
                "comment_permalink": c.get("permalink") or post_url
            })
            dedup_buf.append(body)
            comment_count += 1
                
    except Exception as e:
        logging.warning(f"[comments] Failed to extract comments from {post_url}: {e}")
    
    return solution_comments

def generate_post_summary(problem: str, title: str, text: str, category: str) -> str:
    """
    Generate a concise, informative summary of a Reddit post or comment using OpenAI.
    Focuses ONLY on solutions and actionable steps mentioned.
    Works for both posts and comments.
    """
    if not text or len(text.strip()) < 50:
        return text or title
    
    # Truncate very long posts to avoid token limits
    text_truncated = text[:2000] if len(text) > 2000 else text
    
    prompt = f"""You are extracting the actionable solution from a Reddit post/comment about: {problem}

IMPORTANT RULES:
1. Extract the solution, workaround, recommendation, or actionable steps
2. Include recommendations, tools, methods, or approaches mentioned
3. Focus on WHAT the user did, HOW they fixed it, WHAT they recommend, or WHAT steps they took
4. If the content mentions a tool, website, app, or method that helps solve the problem, include it
5. Only return "No actionable solution found" if there is absolutely NO solution, recommendation, or actionable content
6. Keep it concise (2-3 sentences maximum)
7. Start directly with the solution/recommendation, no preamble

Title: {title}
Content: {text_truncated}

Extract the actionable solution or recommendation. Be lenient - include any tool, method, or approach that could help."""
    
    try:
        resp = get_openai_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[
                {"role": "system", "content": "You are a concise summarizer. Return only the summary text, no labels or prefixes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        summary = resp.choices[0].message.content.strip()
        # Clean up any common prefixes LLM might add
        summary = re.sub(r'^(Summary:|The post|This post|In summary|The user|The comment):\s*', '', summary, flags=re.I)
        return summary if summary else (text[:500] + "..." if len(text) > 500 else text)
    except Exception as e:
        logging.warning(f"[summary] Failed for '{title[:50]}': {e}")
        # Fallback to truncated text
        return text[:500] + "..." if len(text) > 500 else text

def score_competitor_relevance(problem: str, competitor_name: str, competitor_description: str, 
                                how_it_addresses: str, strengths_weaknesses: str) -> float:
    """
    Score how relevant a competitor/solution is to the problem statement (0-100%).
    Uses OpenAI to evaluate relevance based on how well the solution addresses the problem.
    """
    if not problem or not competitor_name:
        return 0.0
    
    # Combine all competitor information
    competitor_info = f"""
Competitor: {competitor_name}
Description: {competitor_description}
How it addresses the problem: {how_it_addresses}
Strengths and weaknesses: {strengths_weaknesses}
"""
    
    prompt = f"""You are evaluating how relevant and useful a competitor/solution is for solving a specific problem.

Problem Statement: {problem}

Competitor Information:
{competitor_info}

Task: Score the relevance of this competitor on a scale of 0-100%, where:
- 100% = Directly and comprehensively addresses the core problem
- 75-99% = Addresses the problem well with some gaps
- 50-74% = Partially addresses the problem or addresses related aspects
- 25-49% = Tangentially related, minimal relevance
- 0-24% = Not relevant or doesn't address the problem

Consider:
1. How directly the solution addresses the stated problem
2. How well it solves the core issue vs. just related aspects
3. The completeness of the solution
4. The quality and effectiveness based on strengths/weaknesses

Return ONLY a JSON object with this exact format:
{{"relevance_score": 85.5, "reason": "brief explanation"}}

The relevance_score should be a number between 0 and 100."""
    
    try:
        resp = get_openai_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[
                {"role": "system", "content": "You are a relevance scorer. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        raw = resp.choices[0].message.content.strip()
        
        # Extract JSON from response
        m = re.search(r'\{.*\}', raw, flags=re.S)
        if m:
            result = json.loads(m.group(0))
            score = float(result.get("relevance_score", 0.0))
            # Clamp to 0-100
            return max(0.0, min(100.0, score))
        return 50.0  # Default to 50% if parsing fails
    except Exception as e:
        logging.warning(f"[scoring] Failed to score competitor '{competitor_name}': {e}")
        return 50.0  # Default to 50% if scoring fails

def score_reddit_relevance(problem: str, title: str, text: str, summary: str = "") -> float:
    """
    Score how relevant a Reddit post/comment is to the problem statement (0-100%).
    Uses a combination of heuristics and OpenAI to calculate relevance score.
    """
    if not problem or not text:
        return 0.0
    
    # Heuristic scoring (0-40 points)
    heuristic_score = 0.0
    base_terms = set(_terms_from(problem))
    problem_lower = problem.lower()
    content = f"{title} {text}".lower()
    summary_lower = (summary or "").lower()
    combined = f"{content} {summary_lower}".lower()
    
    # Term matching (0-20 points)
    if problem_lower in combined:
        heuristic_score += 20.0  # Full phrase match
    elif base_terms:
        hits = sum(1 for t in base_terms if t in combined)
        total_terms = len(base_terms)
        if total_terms > 0:
            term_match_ratio = hits / total_terms
            heuristic_score += term_match_ratio * 20.0
    
    # Solution quality indicators (0-20 points)
    solution_indicators = [
        r'(i\s+(used|tried|fixed|solved|did|implemented|applied|followed|took))',
        r'(what\s+worked\s+(for\s+me|in\s+my\s+case))',
        r'(here\'?s\s+(how|what)\s+(i|to|you))',
        r'(step\s+\d+|step\s+by\s+step|first\s+.*then|next\s+.*finally)',
        r'(try\s+(this|these|using|doing))',
        r'(you\s+can\s+(try|use|do|implement|apply|follow))',
        r'(solution\s+(is|was|to|involves))',
        r'(workaround\s+(is|was|to|involves))',
        r'(i\s+recommend\s+(doing|using|trying|that))',
        r'(give\s+\w+\s+a\s+try)',
    ]
    solution_matches = sum(1 for pattern in solution_indicators if re.search(pattern, combined, re.I))
    heuristic_score += min(20.0, solution_matches * 2.0)  # Up to 20 points for solution indicators
    
    # Use OpenAI for fine-grained scoring (0-60 points) - only if heuristic score is reasonable
    if heuristic_score >= 10.0:  # Only use OpenAI if there's some relevance
        try:
            content_for_scoring = summary if summary and len(summary) > 50 else text[:1000]
            prompt = f"""You are evaluating how relevant and useful a Reddit solution is for solving a specific problem.

Problem Statement: {problem}

Reddit Solution:
Title: {title}
Content: {content_for_scoring}

Task: Score the relevance on a scale of 0-100%, where:
- 100% = Directly solves the exact problem with actionable steps
- 75-99% = Highly relevant solution that addresses the core problem
- 50-74% = Relevant solution that addresses aspects of the problem
- 25-49% = Somewhat relevant but may not fully address the problem
- 0-24% = Not relevant or doesn't address the problem

Return ONLY a JSON object: {{"relevance_score": 85.5, "reason": "brief explanation"}}"""
            
            resp = get_openai_client().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", MODEL),
                messages=[
                    {"role": "system", "content": "You are a relevance scorer. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=100
            )
            raw = resp.choices[0].message.content.strip()
            
            m = re.search(r'\{.*\}', raw, flags=re.S)
            if m:
                result = json.loads(m.group(0))
                llm_score = float(result.get("relevance_score", 0.0))
                # Combine: 40% heuristic + 60% LLM
                final_score = (heuristic_score * 0.4) + (llm_score * 0.6)
                return max(0.0, min(100.0, final_score))
        except Exception as e:
            logging.debug(f"[scoring] LLM scoring failed for Reddit post, using heuristic only: {e}")
    
    # If OpenAI fails or heuristic score is low, use heuristic only (scale to 0-100)
    return max(0.0, min(100.0, heuristic_score * 2.5))  # Scale 0-40 to 0-100

def validate_solution_relevance(problem: str, solution_text: str, solution_type: str = "post") -> bool:
    """
    Validate if a solution is actually relevant and valid for the given problem statement.
    Uses OpenAI to check if the solution addresses the problem.
    Returns True if the solution is valid and relevant, False otherwise.
    """
    if not solution_text or len(solution_text.strip()) < 20:
        return False
    
    # Truncate for API limits
    solution_truncated = solution_text[:1500] if len(solution_text) > 1500 else solution_text
    
    prompt = f"""You are validating whether a solution/workaround is actually relevant and helpful for solving a specific problem.

Problem Statement: {problem}

Solution/Workaround (from Reddit {solution_type}):
{solution_truncated}

Task: Determine if this solution is:
1. Actually relevant to the problem statement
2. A valid/helpful solution that addresses the problem
3. Not just a generic or unrelated piece of advice

Return ONLY a JSON object with this exact format:
{{"is_valid": true/false, "reason": "brief explanation"}}

Return true ONLY if the solution directly addresses the problem. Return false if:
- The solution is unrelated to the problem
- The solution is too generic/vague
- The solution doesn't actually solve the stated problem
- The solution is about a different problem entirely"""
    
    try:
        resp = get_openai_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", MODEL),
            messages=[
                {"role": "system", "content": "You are a solution validator. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        raw = resp.choices[0].message.content.strip()
        
        # Extract JSON from response
        m = re.search(r'\{.*\}', raw, flags=re.S)
        if m:
            result = json.loads(m.group(0))
            is_valid = result.get("is_valid", False)
            reason = result.get("reason", "")
            if not is_valid:
                logging.debug(f"[validation] Solution rejected: {reason[:100]}")
            return is_valid
        return True  # Default to valid if parsing fails (conservative)
    except Exception as e:
        logging.warning(f"[validation] Failed to validate solution: {e}")
        return True  # Default to valid if validation fails (conservative - don't filter out on error)

def compute_frs_pvi(df: pd.DataFrame,
                    F_cap=F_CAP_DEFAULT,
                    H=RECENCY_HALF_LIFE,
                    weights=WEIGHTS) -> Dict:
    N = max(len(df), 1)
    F = 100.0 * min(N / float(F_cap), 1.0)
    R_raw = df["days_ago"].apply(lambda d: math.exp(-d / H)).mean() if N else 0.0
    R = 100.0 * R_raw
    def mean(cat):
        v = df.loc[df["category"] == cat, "sentiment_compound"].mean()
        return 0.0 if pd.isna(v) else float(v)
    seek_m = mean("Solution Seeking"); work_m = mean("Workaround")
    # For solution-only mode, positive sentiment in workarounds is good
    S_raw = clamp((0.7*work_m + 0.3*seek_m), 0.0, 1.0)
    S = 100.0 * S_raw
    wF, wR, wS = weights
    PVI = wF*F + wR*R + wS*S
    rating = "Strong" if PVI >= 70 else ("Moderate" if PVI >= 40 else "Weak")
    return {"N": int(N), "F": round(F,1), "R": round(R,1), "S": round(S,1), "PVI": round(PVI,1), "rating": rating}

def reddit_summary_markdown(problem: str, df: pd.DataFrame, pvi: Dict) -> str:
    if df.empty:
        return "No solution posts or solution comments found in the specified window."
    
    # Separate posts and comments
    posts_df = df[df["source"] == "reddit"].copy()
    comments_df = df[df["source"] == "reddit_comment"].copy() if "source" in df.columns else pd.DataFrame()
    
    total_posts = len(posts_df)
    total_comments = len(comments_df)
    total = total_posts + total_comments
    
    if total == 0:
        return "No solution posts or solution comments found in the specified window."
    
    oldest = utc_str(df["created_utc"].min())
    newest = utc_str(df["created_utc"].max())

    def solution_snips(df_subset, n=6):
        """Show top solutions from posts or comments."""
        if df_subset.empty:
            return []
        tmp = df_subset.copy()
        tmp = tmp.sort_values(["score", "emotional_intensity"], ascending=False).head(n)
        out = []
        for _, r in tmp.iterrows():
            # Prioritize summary - it should contain the extracted solution
            summary = r.get("summary", "").strip()
            if summary and len(summary) > 20:
                # Use summary if it exists and is meaningful
                solution_text = summary
            else:
                # Fallback to short quote (truncated text) if no summary
                solution_text = r.get("short_quote", "") or r.get("text", "")[:300]
            
            # Truncate if too long (but summaries should already be concise)
            if len(solution_text) > 500:
                solution_text = solution_text[:497] + "..."
            
            source_type = "Comment" if r.get("source") == "reddit_comment" else "Post"
            parent_info = ""
            if r.get("source") == "reddit_comment" and r.get("parent_post_title"):
                parent_info = f"\n  *Comment on post: \"{short(r['parent_post_title'], 100)}\"*"
            
            out.append(
                f'- **{source_type}:** {solution_text}'
                f'{parent_info}\n'
                f'  — r/{r["subreddit"]} · {utc_str(r["created_utc"])} · {r["permalink"]}\n'
            )
        return out

    lines = []
    lines.append(f"### Reddit Solutions Found — {problem}")
    lines.append(f"- Window: {oldest} → {newest}")
    lines.append(f"- Count: {total_posts} solution posts, {total_comments} solution comments\n")
    
    lines.append("#### Solutions & Workarounds")
    # Combine all solutions (posts + comments)
    all_solutions = solution_snips(df, n=10)
    lines += (all_solutions or ["(none found)"])
    
    if not comments_df.empty:
        lines.append(f"\n#### Breakdown")
        lines.append(f"- Solution posts: {total_posts}")
        lines.append(f"- Solution comments: {total_comments}")

    return "\n".join(lines)

# =========================
# Main pipeline
# =========================
def run_pipeline(
    problem: str,
    app_description: Optional[str] = None,
    target_user_jtbd: Optional[str] = None,
    geography: Optional[str] = None,
    platforms: Optional[str] = None,
    business_model: Optional[str] = None,
    target_differentiators: Optional[str] = None,
    user_role: Optional[str] = None,
    reddit_query: Optional[str] = None,
    reddit_subreddits: Optional[List[str]] = None,
    reddit_limit: int = 120,      # kept for signature compatibility
    reddit_since_days: int = 365  # (unused; time_filter is set below)
) -> Tuple[Dict[str, Any], Dict[str, Any], str, pd.DataFrame, str]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)

    logging.info("Normalizing problem...")
    p = normalize_problem(problem)

    logging.info("Building solution landscape (LLM-only, inferred)...")
    l = build_landscape(problem)

    logging.info("Running competitive analysis (web-research prompt)...")
    comp_md = run_competitive_analysis(
        problem_statement=problem,  # Pass problem as primary parameter
        app_description=app_description,
        target_user_jtbd=target_user_jtbd,
        geography=geography,
        platforms=platforms,
        business_model=business_model,
        target_differentiators=target_differentiators
    )

    logging.info("Fetching Reddit feedback (SerpAPI + OpenAI, two-pass)...")
    try:
        # scrape (fast defaults; adjust here if needed)
        subs = reddit_subreddits if reddit_subreddits else DEFAULT_SUBS
        rows = scrape_reddit(
            problem=reddit_query or problem,
            subreddits=subs,
            time_filter="month",
            mode="fast",
            per_query_limit=15,  # Increased from 8
            max_per_sub=50  # Increased from 25
        )
        if not rows:
            reddit_df = pd.DataFrame()
            reddit_summary = "No relevant Reddit posts found in the specified window."
        else:
            reddit_df = pd.DataFrame(rows).drop_duplicates(subset=["permalink"]).reset_index(drop=True)
            reddit_df["days_ago"] = reddit_df["created_utc"].apply(days_ago)

            # PASS 1: heuristic
            reddit_df["sentiment_compound"] = [
                heuristic_sentiment(ti, tx) for ti, tx in zip(reddit_df["title"], reddit_df["text"])
            ]
            def _flags(row):
                return _infer_flags(
                    row["title"], row["text"], row["sentiment_compound"],
                    GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS
                )
            reddit_df["is_seek"], reddit_df["is_work"] = zip(*reddit_df.apply(_flags, axis=1))

            def _primary(row):
                # Only Workaround (solution) or Solution Seeking (question that might have solution comments)
                # Prioritize has_solution_in_post flag from initial scrape
                if row.get("has_solution_in_post", False):
                    return "Workaround"
                if row["is_work"]: 
                    return "Workaround"
                if row["is_seek"]: 
                    return "Solution Seeking"
                # If neither, check if it has solution content
                if is_solution_content(row["text"], row["title"]):
                    return "Workaround"
                # Default to Workaround if it's not clearly a question (more lenient)
                # Only mark as Solution Seeking if it's clearly asking a question
                if "?" in (row.get("title", "") + " " + row.get("text", "")).lower():
                    return "Solution Seeking"
                return "Workaround"  # Default to Workaround (more lenient)
            
            reddit_df["category"] = reddit_df.apply(_primary, axis=1)
            logging.info(f"Category breakdown: {reddit_df['category'].value_counts().to_dict()}")
            reddit_df["intent"] = reddit_df["category"].map({"Solution Seeking":"ask","Workaround":"workaround"})
            reddit_df["emotional_intensity"] = [2 + int(3 * abs(s)) for s in reddit_df["sentiment_compound"]]
            reddit_df["short_quote"] = [short(x or y) for x, y in zip(reddit_df["text"], reddit_df["title"])]
            reddit_df["summary"] = ""  # Initialize summary column
            
            # Extract solution comments from ALL posts (not just Solution Seeking)
            # Solution comments can appear in any post, not just question posts
            logging.info("Extracting solution comments from all relevant posts...")
            solution_comments_rows = []
            
            # Extract comments from all posts that are relevant (both Solution Seeking and Workaround posts)
            # This ensures we don't miss solution comments in posts that already contain solutions
            all_relevant_posts = reddit_df.copy()  # Check all posts for solution comments
            
            # Extract base terms once for relevance checking
            base_terms = set(_terms_from(problem))
            
            for _, post_row in all_relevant_posts.iterrows():
                # DOUBLE-CHECK: Verify the post is actually relevant before extracting comments
                post_content = f"{post_row['title']} {post_row['text']}".lower()
                if not _is_relevant_to_problem(post_content, problem, base_terms):
                    logging.info(f"[comments] Skipping irrelevant post: '{post_row['title'][:50]}'")
                    continue
                
                comments = extract_comments_with_solutions(
                    post_row["permalink"], problem, max_comments=15  # Increased from 5
                )
                
                for comment in comments:
                    solution_comments_rows.append({
                        "source": "reddit_comment",
                        "subreddit": post_row["subreddit"],
                        "query": post_row.get("query", ""),
                        "title": f"Comment on: {post_row['title']}",
                        "text": comment["comment_text"],
                        "author": comment["comment_author"],
                        "num_comments": 0,
                        "score": comment["comment_score"],
                        "permalink": comment["comment_permalink"],
                        "created_utc": comment["comment_created_utc"],
                        "days_ago": days_ago(comment["comment_created_utc"]),
                        "parent_post_url": post_row["permalink"],
                        "parent_post_title": post_row["title"],
                        "category": "Workaround",  # Comments with solutions are workarounds
                        "intent": "workaround",
                        "has_solution_in_post": True,
                    })
                time.sleep(0.2)  # Rate limiting for comment fetching
            
            # Filter to only solution posts (Workaround category)
            solution_posts_df = reddit_df[reddit_df["category"] == "Workaround"].copy()
            logging.info(f"Found {len(solution_posts_df)} posts categorized as Workaround (out of {len(reddit_df)} total posts)")
            
            # REMOVED: Final strict filter - it was too aggressive and filtering out valid solutions
            # The initial categorization and has_solution_in_post flag should be sufficient
            # If a post was categorized as Workaround, trust that categorization
            
            # Add solution comments as new rows
            if solution_comments_rows:
                comments_df = pd.DataFrame(solution_comments_rows)
                # Add missing columns to comments_df to match reddit_df
                for col in reddit_df.columns:
                    if col not in comments_df.columns:
                        comments_df[col] = ""
                # Combine solution posts with solution comments
                reddit_df = pd.concat([solution_posts_df, comments_df], ignore_index=True)
                logging.info(f"Found {len(solution_comments_rows)} solution comments from {len(all_relevant_posts)} posts")
            else:
                reddit_df = solution_posts_df
                logging.info("No solution comments found, keeping only solution posts")
            
            if reddit_df.empty:
                reddit_df = pd.DataFrame()
                reddit_summary = "No solution posts or solution comments found in the specified window."
            else:
                # Recalculate sentiment and flags for comments that were added
                reddit_df["sentiment_compound"] = [
                    heuristic_sentiment(ti, tx) for ti, tx in zip(
                        reddit_df["title"].fillna(""), reddit_df["text"].fillna("")
                    )
                ]
                reddit_df["emotional_intensity"] = [2 + int(3 * abs(s)) for s in reddit_df["sentiment_compound"]]
                reddit_df["short_quote"] = [
                    short(x or y) for x, y in zip(reddit_df["text"].fillna(""), reddit_df["title"].fillna(""))
                ]
                
                # Ensure all required columns exist
                if "days_ago" not in reddit_df.columns:
                    reddit_df["days_ago"] = reddit_df["created_utc"].apply(days_ago)
                
                # Ensure category is set for all rows
                if "category" not in reddit_df.columns:
                    reddit_df["category"] = "Workaround"
                if "intent" not in reddit_df.columns:
                    reddit_df["intent"] = "workaround"

            # OpenAI labeling pass (only for solution posts, not comments)
            # Comments are already identified as solutions, skip labeling for them
            posts_only = reddit_df[reddit_df.get("source", "") == "reddit"].copy()
            if not posts_only.empty:
                def _uncertainty(row):
                    amb = int((row.get("is_seek", False) and row.get("is_work", False)) or 
                             (not row.get("is_seek", False) and not row.get("is_work", False)))
                    sent_flat = 1.0 - abs(row.get("sentiment_compound", 0))
                    length_pen = 1.0 if len((row.get("text", "") or "").split()) < 25 else 0.0
                    recency_boost = math.exp(-row.get("days_ago", 0)/RECENCY_HALF_LIFE)
                    return 0.6*amb + 0.3*sent_flat + 0.1*length_pen + 0.2*recency_boost
                posts_only["uncertainty"] = posts_only.apply(_uncertainty, axis=1)
                k = min(24, len(posts_only))  # max LLM calls
                df_llm = posts_only.sort_values("uncertainty", ascending=False).head(k).copy()

                for i, r in df_llm.iterrows():
                    try:
                        res = openai_label(problem, r["title"], r["text"], r["subreddit"],
                                           GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS)
                        llm_sent = float(res.get("sentiment_compound", 0.0))
                        if abs(llm_sent) >= 0.2:
                            reddit_df.at[i, "sentiment_compound"] = llm_sent
                        reddit_df.at[i, "category"] = res["category"]
                        reddit_df.at[i, "intent"] = res["intent"]
                        reddit_df.at[i, "emotional_intensity"] = int(res["emotional_intensity"])
                        reddit_df.at[i, "short_quote"] = res["short_quote"]
                        # re-derive overlaps after LLM
                        s_flag, w_flag = _infer_flags(
                            r["title"], r["text"], reddit_df.at[i, "sentiment_compound"],
                            GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS
                        )
                        reddit_df.at[i,"is_seek"], reddit_df.at[i,"is_work"] = s_flag, w_flag
                    except Exception:
                        pass
                    if i % 5 == 0:
                        time.sleep(0.05)

            # Generate summaries for ALL posts AND comments in final output (prioritize by engagement)
            logging.info("Generating summaries for all solution posts and comments...")
            reddit_df["engagement"] = reddit_df["score"] + reddit_df["num_comments"] * 2
            # Sort by engagement to prioritize important items for summary generation
            reddit_df_sorted = reddit_df.sort_values(
                ["engagement", "emotional_intensity"], ascending=False
            )
            
            summary_count = 0
            for i, r in reddit_df_sorted.iterrows():
                if not r.get("summary") or r.get("summary") == "":  # Only generate if not already set
                    try:
                        # Determine if this is a post or comment
                        is_comment = r.get("source") == "reddit_comment"
                        solution_type = "comment" if is_comment else "post"
                        
                        summary = generate_post_summary(
                            problem, r["title"], r["text"], r["category"]
                        )
                        # Only set summary if it's meaningful (not just fallback text)
                        if summary and len(summary.strip()) > 20:
                            reddit_df.at[i, "summary"] = summary
                            summary_count += 1
                        else:
                            # If summary generation failed or returned empty, use short quote
                            reddit_df.at[i, "summary"] = reddit_df.at[i, "short_quote"]
                        if summary_count % 3 == 0:
                            time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        logging.warning(f"[summary] Failed for {solution_type} at index {i}: {e}")
                        reddit_df.at[i, "summary"] = reddit_df.at[i, "short_quote"]  # Fallback
            
            # Drop the temporary engagement column
            reddit_df = reddit_df.drop(columns=["engagement"], errors='ignore')
            
            # Filter out items that have "No actionable solution found" in summary
            # Only filter if summary column exists and has values
            # BE MORE LENIENT: Only filter if the summary explicitly says "No actionable solution found"
            # Don't filter if summary is empty or just says something else
            if "summary" in reddit_df.columns and not reddit_df.empty:
                before_filter = len(reddit_df)
                # Only filter exact matches - be very strict about what to filter
                mask = reddit_df["summary"].fillna("").str.strip().str.lower() == "no actionable solution found"
                reddit_df = reddit_df[~mask].copy()
                after_filter = len(reddit_df)
                if before_filter != after_filter:
                    logging.info(f"Filtered out {before_filter - after_filter} items with explicit 'No actionable solution found'")
            
            logging.info(f"Generated {summary_count} summaries for solution posts and comments")
            
            # Validate solutions are actually relevant to the problem
            logging.info("Validating solution relevance to problem statement...")
            validation_results = []
            validation_count = 0
            
            # Validate top solutions (prioritize by score and engagement)
            emotional_intensity_col = reddit_df.get("emotional_intensity", pd.Series([0] * len(reddit_df)))
            if isinstance(emotional_intensity_col, pd.Series):
                reddit_df["validation_priority"] = reddit_df["score"] + emotional_intensity_col * 2
            else:
                reddit_df["validation_priority"] = reddit_df["score"]
            reddit_df_sorted_validation = reddit_df.sort_values("validation_priority", ascending=False)
            
            # Validate up to 30 items (to avoid too many API calls)
            max_validation = min(30, len(reddit_df_sorted_validation))
            
            for i, r in reddit_df_sorted_validation.head(max_validation).iterrows():
                try:
                    is_comment = r.get("source") == "reddit_comment"
                    solution_type = "comment" if is_comment else "post"
                    
                    # Use summary if available, otherwise use text
                    solution_text = r.get("summary", "").strip()
                    if not solution_text or len(solution_text) < 20:
                        solution_text = r.get("text", "").strip()
                    
                    if solution_text and len(solution_text) >= 20:
                        is_valid = validate_solution_relevance(problem, solution_text, solution_type)
                        validation_results.append((i, is_valid))
                        validation_count += 1
                        
                        if not is_valid:
                            # Mark as invalid (we'll filter later)
                            reddit_df.at[i, "is_valid_solution"] = False
                        else:
                            reddit_df.at[i, "is_valid_solution"] = True
                        
                        if validation_count % 5 == 0:
                            time.sleep(0.2)  # Rate limiting for validation
                except Exception as e:
                    logging.warning(f"[validation] Failed for item at index {i}: {e}")
                    # Default to valid if validation fails
                    reddit_df.at[i, "is_valid_solution"] = True
            
            # Mark unvalidated items as valid (conservative - don't filter out unvalidated items)
            if "is_valid_solution" not in reddit_df.columns:
                reddit_df["is_valid_solution"] = True
            else:
                # Use infer_objects to avoid FutureWarning about downcasting
                reddit_df["is_valid_solution"] = reddit_df["is_valid_solution"].fillna(True).infer_objects(copy=False)
            
            # Filter out invalid solutions (only if validation was performed)
            if validation_count > 0 and "is_valid_solution" in reddit_df.columns:
                before_validation_filter = len(reddit_df)
                # Use fillna to handle any NaN values, then filter
                reddit_df["is_valid_solution"] = reddit_df["is_valid_solution"].fillna(True)
                reddit_df = reddit_df[reddit_df["is_valid_solution"] == True].copy()
                after_validation_filter = len(reddit_df)
                
                invalid_count = before_validation_filter - after_validation_filter
                if invalid_count > 0:
                    logging.info(f"Filtered out {invalid_count} solutions that were not relevant to the problem")
            
            # Calculate relevance scores for Reddit posts/comments
            logging.info("Calculating relevance scores for Reddit posts and comments...")
            reddit_df["relevance_score"] = 0.0
            
            # Score all items
            scored_count = 0
            for i, r in reddit_df.iterrows():
                try:
                    title = r.get("title", "")
                    text = r.get("text", "")
                    summary = r.get("summary", "")
                    
                    # Use summary if available for better scoring
                    score = score_reddit_relevance(problem, title, text, summary)
                    reddit_df.at[i, "relevance_score"] = round(score, 1)
                    scored_count += 1
                    
                    # Rate limiting for scoring
                    if scored_count % 10 == 0:
                        time.sleep(0.1)
                except Exception as e:
                    logging.warning(f"[scoring] Failed to score Reddit item at index {i}: {e}")
                    reddit_df.at[i, "relevance_score"] = 50.0  # Default score
            
            logging.info(f"Calculated relevance scores for {scored_count} Reddit items")
            
            # Drop temporary validation columns
            reddit_df = reddit_df.drop(columns=["validation_priority", "is_valid_solution"], errors='ignore')
            
            if validation_count > 0:
                logging.info(f"Validated {validation_count} solutions, {len(reddit_df)} remain after filtering")

            # Compute PVI only if we have data (for solution-only mode, PVI is less relevant)
            if not reddit_df.empty:
                try:
                    pvi = compute_frs_pvi(reddit_df)
                except Exception:
                    # Fallback PVI if calculation fails
                    pvi = {"N": len(reddit_df), "F": 50.0, "R": 50.0, "S": 50.0, "PVI": 50.0, "rating": "Moderate"}
            else:
                pvi = {"N": 0, "F": 0.0, "R": 0.0, "S": 0.0, "PVI": 0.0, "rating": "Weak"}
            
            reddit_summary = reddit_summary_markdown(problem, reddit_df, pvi)

    except Exception as e:
        logging.error(f"Reddit fetch/label failed: {e}")
        reddit_df = pd.DataFrame()
        reddit_summary = f"Reddit fetch/label failed: {e}"

    return p, l, comp_md, reddit_df, reddit_summary

def save_outputs(
    p: Dict[str, Any],
    l: Dict[str, Any],
    comp_md: str,
    reddit_df: pd.DataFrame,
    reddit_summary: str
) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save LLM-only JSON + MD
    with open(OUT_JSON_LLM, "w", encoding="utf-8") as f:
        json.dump({"problem": p, "analysis": l}, f, ensure_ascii=False, indent=2)
    with open(OUT_MD_LLM, "w", encoding="utf-8") as f:
        f.write(to_markdown(p, l))

    # Save Competitive analysis markdown + Reddit Summary
    with open(OUT_MD_COMP, "w", encoding="utf-8") as f:
        f.write(comp_md)
        f.write("\n\n---\n## Reddit Summary (LLM)\n")
        f.write(reddit_summary)

    with open(OUT_JSON_COMP, "w", encoding="utf-8") as f:
        json.dump({"competitive_analysis_markdown": comp_md, "reddit_summary": reddit_summary}, f, ensure_ascii=False, indent=2)

    # Save Reddit CSV
    if reddit_df is not None and not reddit_df.empty:
        reddit_df.to_csv(OUT_REDDIT_CSV, index=False, encoding="utf-8")

    # One Excel workbook with all relevant tabs
    to_excel(l, comp_md, reddit_df)

# =========================
# Example run
# =========================
def run_m4(problem_statement: str, user_role: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for Module 4 API.
    Takes a problem statement and optional user role, returns the analysis results.
    """
    try:
        # Run the pipeline (only problem statement needed)
        p, l, comp_md, reddit_df, reddit_summary = run_pipeline(
            problem=problem_statement,
            user_role=user_role,
            reddit_query=None,  # Optional
            reddit_subreddits=None,  # Optional - uses default subreddits
        )
        
        # Save outputs
        save_outputs(p, l, comp_md, reddit_df, reddit_summary)
        
        # Convert DataFrame to dict for JSON serialization
        reddit_data = None
        if reddit_df is not None and not reddit_df.empty:
            # Convert DataFrame to dict, handling non-serializable types
            # Replace NaN with None and convert datetime/timestamp columns
            reddit_df_copy = reddit_df.copy()
            # Convert timestamp columns to strings
            if 'created_utc' in reddit_df_copy.columns:
                reddit_df_copy['created_utc'] = reddit_df_copy['created_utc'].astype(float)
            if 'days_ago' in reddit_df_copy.columns:
                reddit_df_copy['days_ago'] = reddit_df_copy['days_ago'].astype(float)
            # Fill NaN with None for JSON serialization
            reddit_df_copy = reddit_df_copy.fillna("")
            reddit_data = reddit_df_copy.to_dict(orient="records")
        
        # Parse competitive analysis into structured format
        competitive_analysis_structured = parse_competitive_analysis_structured(comp_md)
        competitor_count = competitive_analysis_structured.get("total_count", 0)
        
        # Add relevance scores to competitors
        logging.info("Calculating relevance scores for competitors...")
        for cat in competitive_analysis_structured.get("categories", []):
            for sol in cat.get("solutions", []):
                try:
                    score = score_competitor_relevance(
                        problem_statement,
                        sol.get("name", ""),
                        sol.get("description", ""),
                        sol.get("how_it_addresses_problem", ""),
                        sol.get("strengths_and_weaknesses", "")
                    )
                    sol["relevance_score"] = round(score, 1)
                except Exception as e:
                    logging.warning(f"[scoring] Failed to score competitor '{sol.get('name', '')}': {e}")
                    sol["relevance_score"] = 50.0  # Default score
        
        # Also score flat competitor list for backward compatibility
        for comp in competitive_analysis_structured.get("competitors", []):
            if "relevance_score" not in comp:
                try:
                    score = score_competitor_relevance(
                        problem_statement,
                        comp.get("name", ""),
                        comp.get("description", ""),
                        comp.get("how_it_addresses_problem", ""),
                        comp.get("strengths_and_weaknesses", "")
                    )
                    comp["relevance_score"] = round(score, 1)
                except Exception as e:
                    logging.warning(f"[scoring] Failed to score competitor '{comp.get('name', '')}': {e}")
                    comp["relevance_score"] = 50.0  # Default score
        
        return {
            "problem": p,
            "landscape": l,
            "competitive_analysis": {
                "structured": competitive_analysis_structured,
                "markdown": comp_md,  # Keep full markdown for reference
                "competitor_count": competitor_count  # Easy access for UI display
            },
            "reddit_summary": reddit_summary,
            "reddit_data": reddit_data,
            "citations": citations, # Add citations here
            "output_files": {
                "json_llm": OUT_JSON_LLM,
                "md_llm": OUT_MD_LLM,
                "md_competitive": OUT_MD_COMP,
                "json_competitive": OUT_JSON_COMP,
                "reddit_csv": OUT_REDDIT_CSV,
                "excel": OUT_XLSX
            }
        }
    except Exception as e:
        logging.error(f"Error in run_m4: {e}")
        raise

if __name__ == "__main__":
    problem_statement = (
        "The current hiring process creates a destructive cycle where candidates feel compelled to game the "
        "system while companies struggle to see past the noise."
    )

    # Now you can call with just the problem statement!
    # All other parameters are optional
    p, l, comp_md, reddit_df, reddit_summary = run_pipeline(
        problem=problem_statement,
        # Optional: provide additional context if you have an app
        # app_description="...",
        # target_user_jtbd="...",
        # etc.
        reddit_query="hiring assessments cheating proctoring resume signal noise interview platform",
        reddit_subreddits=["recruitinghell","humanresources","cscareerquestions","jobs","career_advice"],
    )

    save_outputs(p, l, comp_md, reddit_df, reddit_summary)
    print("\nDone!")
    print(f"- LLM-only JSON: {OUT_JSON_LLM}")
    print(f"- LLM-only Markdown: {OUT_MD_LLM}")
    print(f"- Competitive Analysis Markdown: {OUT_MD_COMP}")
    print(f"- Competitive Analysis JSON: {OUT_JSON_COMP}")
    print(f"- Reddit CSV: {OUT_REDDIT_CSV}")
    print(f"- Excel Workbook: {OUT_XLSX}")

# =========================
# Adapter for StudioOS Flask Backend
# =========================

def run_m4(problem: str, api_key: str = None) -> dict:
    """
    Main function for M4 module (StudioOS Adapter).
    Executes the competitive analysis using the latest logic.
    """
    try:
        # Configure OpenAI client with dynamic key
        global OPENAI_API_KEY, client
        if api_key:
            OPENAI_API_KEY = api_key
            client = OpenAI(api_key=api_key)
        
        # Run the analysis
        # We'll use the problem statement as the input for the competitive analysis
        analysis_md = run_competitive_analysis(
            problem_statement=problem,
            # We can optionally infer other params or leave them null to use the problem-only mode
        )
        
        # Parse the markdown to structured data if possible, or just return the markdown
        structured_data = parse_competitive_analysis_structured(analysis_md)
        
        # Extract citations from competitors
        citations = []
        if structured_data and "competitors" in structured_data:
            for comp in structured_data["competitors"]:
                if comp.get("url"):
                    citations.append({
                        "title": comp.get("name", "Competitor"),
                        "uri": comp.get("url"),
                        "source": "Web"
                    })

        return {
            "analysis": analysis_md,
            "structured_analysis": structured_data,
            "citations": citations,
            "problem": problem,
            "model": MODEL,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error in run_m4: {e}", exc_info=True)
        return {
            "error": str(e),
            "analysis": f"Error generating analysis: {str(e)}",
            "problem": problem
        }
