"""
Module M6: Market Analysis & Competitive Intelligence
Refactored for FastAPI integration + AlphaVantage enrichment
contributor commit test by Faye
"""

import sys
import os
import re
import json
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from datetime import date

# ---------------------------------------------------------------------------
# Environment & Path
# ---------------------------------------------------------------------------
load_dotenv()

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
studio_dir = os.path.join(grandparent_dir, "studio-chain-flow-main")
sys.path.insert(0, studio_dir)
sys.path.insert(0, parent_dir)

# ---------------------------------------------------------------------------
# API keys - News & OpenAI & AlphaVantage
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# NewsAPI helpers
# ---------------------------------------------------------------------------

def fetch_news_single_query(query: str) -> List[dict]:
    """
    Fetch news using NewsAPI (short query).
    """
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={quote_plus(query)}&"
        "language=en&"
        "sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            print("❌ NewsAPI error:", data)
            return []

        return [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "ts": a.get("publishedAt"),
                "source": a.get("source", {}).get("name"),
                "description": a.get("description"),
            }
            for a in data.get("articles", [])[:5]
        ]

    except Exception as e:
        print("⚠ fetch_news_single_query error:", e)
        return []


def fetch_news_multi(queries: List[str], max_total: int = 10) -> List[dict]:
    all_articles: List[dict] = []
    seen_urls = set()

    for q in queries or []:
        q = str(q).strip()
        if not q:
            continue

        articles = fetch_news_single_query(q)
        for art in articles:
            url = art.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_articles.append(art)
            if len(all_articles) >= max_total:
                break
        if len(all_articles) >= max_total:
            break

    return all_articles

# ---------------------------------------------------------------------------
# Step 0: topic → market_space + search_queries
# ---------------------------------------------------------------------------

def extract_market_and_queries(problem: str) -> Dict[str, Any]:
    """
    Use OpenAI to turn a long problem description into:
    - market_space
    - search_queries
    - themes
    """
    # direct use short prompt
    if len(problem.split()) < 10:
        return {
            "market_space": problem.strip(),
            "search_queries": [problem.strip()],
            "themes": []
        }

    system_prompt = """You turn long problem descriptions into market intelligence metadata.
Return strictly valid JSON with:
{
  "market_space": "...",
  "search_queries": ["...", "..."],
  "themes": ["...", "..."]
}
"""

    user_prompt = f"Problem:\n{problem}\n\nReturn ONLY JSON."

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)

        # protection
        market_space = data.get("market_space") or problem[:120]
        raw_queries = data.get("search_queries")
        if isinstance(raw_queries, str):
            search_queries = [raw_queries]
        elif isinstance(raw_queries, list):
            search_queries = [
                str(x).strip() for x in raw_queries
                if str(x).strip()
            ]
        else:
            search_queries = [market_space]

        themes_raw = data.get("themes")
        if isinstance(themes_raw, list):
            themes = [str(x).strip() for x in themes_raw if str(x).strip()]
        else:
            themes = []

        if not search_queries:
            search_queries = [market_space]

        return {
            "market_space": market_space,
            "search_queries": search_queries,
            "themes": themes,
        }

    except Exception as e:
        print("⚠ extract_market_and_queries:", e)
        return {
            "market_space": problem[:120],
            "search_queries": [problem[:120]],
            "themes": []
        }

# ---------------------------------------------------------------------------
# AlphaVantage Helpers（Ticker + overviews + Earnings Call）
# ---------------------------------------------------------------------------

def infer_ticker_from_name(name: str) -> Optional[str]:
    """
    Try to parse ticker from a competitor name like 'Microsoft (MSFT)'.
    """
    if not name:
        return None
    m = re.search(r"\(([A-Z\.\-]{1,10})\)", name)
    if m:
        return m.group(1)
    return None


def fetch_company_overview(symbol: str) -> Optional[dict]:
    """
    AlphaVantage Company Overview.
    """
    if not symbol:
        return None

    url = (
        "https://www.alphavantage.co/query?"
        f"function=OVERVIEW&"
        f"symbol={symbol}&"
        f"apikey={ALPHAVANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        # 防止 Note / Information 等错误情况
        if not isinstance(data, dict) or "Symbol" not in data:
            return None
        return data
    except Exception as e:
        print(f"⚠ fetch_company_overview error for {symbol}: {e}")
        return None


def guess_latest_quarter(today: Optional[date] = None) -> str:
    """
    返回最近一个“已经结束”的财季，格式 YYYYQn
    比如现在是 2025-11 月，就返回 2025Q3 （避免请求还没发布的 2025Q4）
    """
    if today is None:
        today = date.today()

    q = (today.month - 1) // 3 + 1  # 当前季度 1-4

    # 为了保险，用上一个季度
    if q == 1:
        year = today.year - 1
        q_num = 4
    else:
        year = today.year
        q_num = q - 1

    return f"{year}Q{q_num}"


def fetch_earnings_call(symbol: str) -> Optional[dict]:
    """
    AlphaVantage Earnings Call Transcript (latest finished quarter).
    """
    if not symbol:
        return None

    quarter = guess_latest_quarter()
    url = (
        "https://www.alphavantage.co/query?"
        f"function=EARNINGS_CALL_TRANSCRIPT&"
        f"symbol={symbol}&"
        f"quarter={quarter}&"
        f"apikey={ALPHAVANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not isinstance(data, dict):
            return None
        if any(k in data for k in ("Note", "Information", "Error Message")):
            return None

        return data
    except Exception as e:
        print(f"⚠ fetch_earnings_call error for {symbol}: {e}")
        return None


def enrich_competitor_profiles_with_alpha(
    competitor_profiles: List[dict],
    max_competitors: int = 10,
) -> List[dict]:
    enriched: List[dict] = []

    important_keys = [
        "Symbol", "Name", "Country", "Sector", "Industry",
        "OfficialSite", "MarketCapitalization", "RevenueTTM",
        "EBITDA", "ProfitMargin", "ReturnOnEquityTTM", "Beta", "52WeekHigh", "52WeekLow",
    ]

    for idx, profile in enumerate(competitor_profiles):
        if idx >= max_competitors:
            enriched.extend(competitor_profiles[idx:])
            break

        profile = dict(profile)
        ticker = profile.get("ticker") or infer_ticker_from_name(profile.get("name", ""))
        overview = fetch_company_overview(ticker) if ticker else None

        if overview:
            profile["ticker"] = ticker
            # 原始全量：调试 / 以后要加字段可以用
            profile["alpha_overview"] = {
                k: overview.get(k)
                for k in important_keys
                if k in overview
            }

        enriched.append(profile)

    return enriched


def collect_earnings_snippets(
    profiles: List[dict],
    max_companies: int = 10,
    max_chars_per: int = 4000,
    total_char_limit: int = 16000
) -> str:
    """
    Pull the earnings call snippets of several companies for LLM extraction.
    Always coerce the transcript into a string to avoid `.strip()` on list/dict.
    """

    snippets: List[str] = []
    total_chars = 0

    for profile in profiles:
        if len(snippets) >= max_companies:
            break
        ticker = profile.get("ticker")
        if not ticker:
            continue

        data = fetch_earnings_call(ticker)
        if not data:
            continue

        raw = ""

        if isinstance(data, dict):
            transcript = data.get("transcript") or data.get("content") or data.get("summary", "")
            # transcript 可能是 list / dict / str，统一转成 str
            if isinstance(transcript, list):
                # 尝试把 list 里的 dict 拼起来
                try:
                    raw = "\n".join(
                        f"{(item.get('speaker') or '').strip()}: {(item.get('content') or '').strip()}"
                        if isinstance(item, dict) else str(item)
                        for item in transcript
                    )
                except Exception:
                    raw = json.dumps(transcript)
            elif isinstance(transcript, (dict,)):
                raw = json.dumps(transcript)
            else:
                raw = str(transcript)
        else:
            # data 本身不是 dict，直接 dump 掉
            raw = json.dumps(data)

        # 安全兜底：确保 raw 一定是 str
        if not isinstance(raw, str):
            raw = json.dumps(raw)

        raw = raw.strip()
        if not raw:
            continue

        snippet = raw[:max_chars_per]
        tagged = f"Ticker {ticker} ({profile.get('name','')}):\n{snippet}"
        snippets.append(tagged)
        total_chars += len(snippet)
        if total_chars >= total_char_limit:
            break

    return "\n\n".join(snippets)

# ---------------------------------------------------------------------------
# News + Earnings → barriers & historical patterns
# ---------------------------------------------------------------------------

def extract_historical_patterns_smart(
    topic: str,
    news_articles: List[dict],
    earnings_snippets: str,
    max_articles: int = 10,
) -> Dict[str, Any]:
    """
    One-shot:
    - Format news_articles → text block
    - Combine with earnings_snippets
    - Extract barriers + historical patterns
    """

    # ------------------------------
    # Step 1: Build news context
    # ------------------------------
    lines = []
    for art in news_articles[:max_articles]:
        title = art.get("title", "")
        source = art.get("source", "")
        ts = art.get("ts", "")
        url = art.get("url", "")
        desc = art.get("description") or ""

        block = f"- {title} [{source} | {ts}] {url}"
        if desc:
            block += f"\n  Summary: {desc}"

        lines.append(block)

    news_context = "\n".join(lines)

    # ------------------------------
    # Step 2: Call LLM
    # ------------------------------
    system_prompt = """
You are a strategy analyst focusing on barriers to entry and patterns of
success/failure in a given domain.

You will be given:
- A topic (problem framing)
- A set of recent news headlines (with summaries)
- A set of earnings call transcript snippets

Extract:

1) barriers_to_entry_extra:
   - 5-10 concise bullets about structural barriers:
     regulation, capital requirements, data moats, distribution power, etc.

2) historical_patterns:
   - 5-12 concise cases/patterns (successes or failures).
   - Each item must include:
        category (from the given list),
        summary,
        2-6 details,
        source_type ("news" | "earnings_call" | "mixed"),
        source_ref (short ref like company or headline)

Return STRICT JSON:

{
  "barriers_to_entry_extra": ["..."],
  "historical_patterns": [
    {
      "category": "...",
      "summary": "...",
      "details": ["...", "..."],
      "source_type": "news | earnings_call | mixed",
      "source_ref": "..."
    }
  ]
}
""".strip()

    user_prompt = f"""
Topic:
{topic}

News context:
{news_context}

Earnings call snippets:
{earnings_snippets}

Return ONLY JSON.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)

        return {
            "barriers_to_entry_extra": data.get("barriers_to_entry_extra", []),
            "historical_patterns": data.get("historical_patterns", []),
        }

    except Exception as e:
        print("⚠ extract_historical_patterns_smart error:", e)
        return {
            "barriers_to_entry_extra": [],
            "historical_patterns": [],
        }



# ---------------------------------------------------------------------------
# OpenAI main：market_space / cohorts / swot ,etc
# ---------------------------------------------------------------------------

def call_openai(topic: str, headlines: List[str]) -> Dict[str, Any]:
    """
    Ask OpenAI to identify market space, cohorts, competitors, trends, SWOT,
    aligned with GoodFutures + StudioOS v1 context.
    """

    system_prompt = f"""
You are a senior market & competitive intelligence analyst embedded in
GoodFutures' StudioOS v1, an internal AI-powered automation platform.

You MUST align your analysis with this product context:
Product Name: StudioOS v1

Purpose:
StudioOS v1 is an internal AI-powered automation platform designed
to significantly accelerate and enhance the venture ideation, research,
validation, and initial planning phases for GoodFutures. It aims to
systematically discover, analyze, and outline high-potential venture
opportunities aligned with GoodFutures' investment theses
(Workforce Development, Empowerment, and Distribution) and its focus on
AI's impact on the workforce.

Strategic Goal:
To enable GoodFutures to more rapidly and efficiently identify,
de-risk, and launch impactful ventures, increasing the studio's capacity
and improving the quality of its deal flow and NewCo formation.

Target Users:
- Studio Partners / Leadership
- GoodFutures Founders / Entrepreneurs-in-Residence (EIRs)
- Venture Analysts / Researchers

Your job is to:
1. Identify the correct MARKET SPACE at 3 abstraction layers:
   - layer1: broad industry category
   - layer2: thematic sub-category
   - layer3: specific actionable market segment
   The market space should reflect GoodFutures' focus on:
   - Workforce Development
   - Empowerment
   - Distribution
   - AI's impact on the workforce and operations

   Provide a qualitative market sizing using TAM/SAM/SOM with comments and assumptions.
   No need for precise numbers; ranges and directional statements are enough.

2. Group competitors, trends, and signals into COHORTS.
   Use the following cohort taxonomy when relevant:
   - AI Talent Development & Upskilling
   - Cloud + AI Infrastructure
   - IT Modernization & Digital Transformation
   - AI Automation & Security
   - Model Skills / RAG / LangChain Ecosystem
   - Enterprise AI Tooling Ecosystem

3. For each cohort, list:
   - 6–10 competitors (include both direct and adjacent competitors)
   - 3–5 trends
   - 2–4 signals
   Try to assign each signal to the single most relevant cohort.

4. Produce a SWOT analysis for a NEW ENTRANT venture that StudioOS
   might help create in this space (not for existing incumbents).

5. VERY IMPORTANT for downstream data enrichment:
   - When you mention a *publicly listed* competitor, append its main ticker
     in UPPERCASE in parentheses, for example:
       - "Microsoft (MSFT)"
       - "Alphabet (GOOGL)"
       - "Snowflake (SNOW)"
   - Do this consistently in BOTH the `cohorts[].competitors` list and the
     `competitor_profiles[].name` field whenever you can reasonably infer
     the ticker. If you are not sure, just leave the name without ticker.

You MUST follow this EXACT output schema.
All fields are REQUIRED and output is INVALID if any field is missing.

REQUIRED SCHEMA (JSON object):
{{
  "market_space": {{
    "layer1": "string",
    "layer2": "string",
    "layer3": "string"
  }},
  "cohorts": [
    {{
      "name": "string",
      "competitors": ["string"],
      "trends": ["string"],
      "signals": [
        {{
          "title": "string",
          "url": "string",
          "source": "string",
          "ts": "string"
        }}
      ]
    }}
  ],
  "competitor_profiles": [
    {{
      "name": "string",
      "type": "string",
      "geo_focus": "string",
      "products": "string",
      "technology": "string",
      "business_model": "string",
      "funding_insight": "string",
      "team_insight": "string",
      "positioning": "string",
      "strengths": ["string"],
      "weaknesses": ["string"]
    }}
  ],
  "market_segmentation": [
    {{
      "segment_name": "string",
      "description": "string",
      "customer_profile": "string",
      "sizing_comment": "string"
    }}
  ],
  "barriers_to_entry": ["string"],
  "whitespace_opportunities": ["string"],
  "swot": {{
    "strengths": ["string"],
    "weaknesses": ["string"],
    "opportunities": ["string"],
    "threats": ["string"]
  }}
}}

MANDATORY RULES:
- `cohorts` MUST NOT be empty.
- At least ONE cohort MUST be created.
- Every cohort MUST contain at least 1 competitor, 1 trend, and 1 signal.
- If signals are ambiguous, pick the best-fit signal from provided news.
""".strip()

    news_text = "\n".join(headlines) if headlines else "No recent news found."

    user_prompt = f"""
Topic / problem description (from StudioOS pipeline):
{topic}

Recent Headlines (signals for this space):
{news_text}

Remember:
- Focus on venture opportunity framing for GoodFutures / StudioOS.
- Prefer cohorts and competitors that are relevant to workforce and AI-enabled operations.
- Return STRICT JSON only, no commentary.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        ai_data = json.loads(resp.choices[0].message.content)

        # ---------- Cohort Aggregation: competitors + trends ----------
        cohorts = ai_data.get("cohorts", [])
        flat_competitors: List[str] = []
        flat_trends: List[str] = []

        for cohort in cohorts:
            flat_competitors.extend(cohort.get("competitors", []))
            flat_trends.extend(cohort.get("trends", []))

        def dedupe(seq: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        ai_data["competitors"] = dedupe(flat_competitors)
        ai_data["trends"] = dedupe(flat_trends)

        return ai_data

    except Exception as e:
        print(f"⚠ OpenAI parse error: {e}")
        return {
            "market_space": {
                "layer1": topic,
                "layer2": topic,
                "layer3": topic
            },
            "cohorts": [],
            "competitors": [],
            "trends": [],
            "competitor_profiles": [],
            "market_segmentation": [],
            "barriers_to_entry": [],
            "whitespace_opportunities": [],
            "swot": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }
        }

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def analyze_market(topic: str) -> Dict[str, Any]:
    """Main analysis pipeline."""
    # 1) market space & search query
    market_info = extract_market_and_queries(topic)

    # 2) 抓新闻（作为 signals + grounded companies 的原料）
    news = fetch_news_multi(market_info["search_queries"], max_total=10)
    headlines_blocks = [
        f"- {n['title']} ({n['source']} | {n['ts']} | {n['url']})"
        for n in news if n.get("title")
    ]

    # 3) 用 OpenAI 做高层 framing（market_space / cohorts / swot ...）
    ai_data = call_openai(market_info["market_space"], headlines_blocks)

    # --------------------------------------------------------------
    # ⭐ NEW LOGIC: Alpha enrichment directly on LLM competitor profiles
    # --------------------------------------------------------------

    ai_profiles = ai_data.get("competitor_profiles", [])

    # Step 1: LLM profiles → extract ticker if in name (e.g., Microsoft (MSFT))
    for p in ai_profiles:
        name = p.get("name", "")
        ticker = infer_ticker_from_name(name)
        if ticker:
            p["ticker"] = ticker

    # Step 2: Alpha enrich directly
    alpha_enriched_profiles = enrich_competitor_profiles_with_alpha(
        ai_profiles,
        max_competitors=10
    )

    competitor_profiles = alpha_enriched_profiles
    
    # 5) Earnings + News → historical cases & extra barriers
    earnings_snippets = collect_earnings_snippets(competitor_profiles)

    patterns = extract_historical_patterns_smart(
        topic=topic,
        news_articles=news,
        earnings_snippets=earnings_snippets,
        max_articles=10,
    )

    # 6) merge barriers
    def merge_unique(base: List[str], extra: List[str]) -> List[str]:
        base = list(base or [])
        for x in extra or []:
            if x not in base:
                base.append(x)
        return base

    merged_barriers = merge_unique(
        ai_data.get("barriers_to_entry", []),
        patterns.get("barriers_to_entry_extra", []),
    )

    historical_patterns = patterns.get("historical_patterns", [])

    return {
        "report_type": "Market & Competitive Intelligence",
        "topic": topic,
        "market_space": ai_data.get("market_space", market_info["market_space"]),
        "market_segmentation": ai_data.get("market_segmentation", []),
        "swot": ai_data.get("swot", {}),
        # ✅ competitor_profiles 完全是 grounded + Alpha enrich
        "competitor_profiles": competitor_profiles,
        "cohorts": ai_data.get("cohorts", []),
        "barriers_to_entry": merged_barriers,
        "historical_patterns": historical_patterns,
        "whitespace_opportunities": ai_data.get("whitespace_opportunities", []),
        "citations": news[:5],
        "meta": {
            "search_queries": market_info.get("search_queries", []),
            "themes": market_info.get("themes", []),
        },
    }

# ---------------------------------------------------------------------------
# API Entry
# ---------------------------------------------------------------------------
def run_m6(problem: str):
    """Entry point for FastAPI / Railway integration."""
    try:
        topic = problem.strip()
        return analyze_market(topic)
    except Exception as e:
        return {
            "error": str(e),
            "topic": problem,
            "report_type": "Market & Competitive Intelligence",
        }
