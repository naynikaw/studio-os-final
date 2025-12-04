#!/usr/bin/env python3
# m2_serp_lite_final.py
"""
StudioOS — M2 Problem Validation (SerpApi + Reddit .json), minimal flags (polished)

Args:
  --problem "<text>"      : problem statement (required)
  --max-results N         : total reddit threads to pull (default 10)

What it does:
- SerpApi Google search for "<problem> reddit" (+ smart seeds, site:reddit.com)
- Fetch each thread's Reddit .json (post + comments)
- Light quality filter on posts; comments lightly filtered too (length + score)
- Heuristic pass → LLM pass only on the most-uncertain items
- Simple near-duplicate pruning for comments within a thread
- Per-thread dampening so one thread doesn’t dominate
- Console report (no files)

Deps:
  pip install requests pandas tqdm python-dateutil openai packaging

Run:
python m2.py --problem "how to stay productive working remote" --max-results 5
"""

import argparse, json, math, re, time, random, os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
from tqdm import tqdm
from dateutil import tz
import io
import contextlib
from openai import OpenAI  # ← OpenAI SDK

# ========= KEYS (fill these) =========
SERPAPI_KEY      = os.getenv("SERPAPI_KEY", "c02ff4b7a167f3f52916715952e88f8e9f2359a7e915c44f941cace50df7a5d7")      # https://serpapi.com/
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")   # https://platform.openai.com/
OPENAI_MODEL     = "gpt-5.1"                     # You can switch models here
# =====================================

# ---------------- Defaults ----------------
DEFAULT_SUBS = ["AskReddit", "explainlikeimfive"]
F_CAP_DEFAULT = 50
RECENCY_HALF_LIFE = 365
WEIGHTS = (0.4, 0.3, 0.3)
TOTAL_HARD_CAP = 100

# Missing constants
MAX_DAYS_OLD = 365 * 2
COMMENTS_PER_THREAD = 15
MIN_COMMENT_WORDS = 4
MIN_COMMENT_SCORE = 2
MAX_LLM = 24

# Generic triggers
GENERIC_QUESTION_TRIGGERS = (
    " how ", "how do ", "how can ", " any tips", " tips ", "recommend", "recommendation",
    "suggest", "suggestion", "what should ", "best way", "advice", "any idea", "help?",
    " how?", " where ", " where?", " which ", " which?", " problem?", " fix?"
)
GENERIC_WORKAROUND_TRIGGERS = (
    "i used", "i tried", "i fixed", "i solved", "what worked for me", "i do this", "my workaround",
    "step by step", "steps i took", "script", "macro", "automation", "template", "checklist", "guide"
)
GENERIC_PAIN_TRIGGERS = (
    "hate", "annoy", "angry", "upset", "frustrat", "sucks", "blocked", "stuck", "broken",
    "bug", "error", "issue", "problem", "fail", "failed", "not working", "doesn't work",
    "didn't work", "worse", "hard", "pain", "struggl", "confus", "overwhelmed"
)

# Aliases for compatibility with local code
WORK_PATTERNS = GENERIC_WORKAROUND_TRIGGERS
ASK_PATTERNS = GENERIC_QUESTION_TRIGGERS

SENT_NEG = {
    "hate","annoy","angry","upset","frustrat","sucks","terrible","awful","bad","pain","hurts",
    "broken","bug","error","issue","problem","fail","failed","worse","hard","stuck","blocked",
    "confusing","confused","overwhelmed","anxious","worried","panic","stress","stressed","rant"
}
SENT_POS = {
    "love","amazing","great","awesome","good","works","fixed","helped","improved",
    "clear","better","success","win","happy","nice","recommend","useful","effective"
}

# --------------- Helpers --------------------
def days_ago(ts: float) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days

def clamp(v, a, b): return max(a, min(b, v))

def short(s, n=220):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n-1] + "…"

# ... (lines 44-382 unchanged)

class _OpenAIAdapter:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        self.chat = client.chat

def init_gemini():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        raise SystemExit("Fill OPENAI_API_KEY at top or set env OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # Use standard OpenAI endpoint
    client = OpenAI(api_key=OPENAI_API_KEY)
    return _OpenAIAdapter(client, OPENAI_MODEL)

CLS_PROMPT = """Classify a short social post about a problem theme.
Return STRICT JSON (one object):
{
  "category": "Pain" | "Solution Seeking" | "Workaround",
  "intent": "complaint" | "ask" | "workaround",
  "sentiment_compound": float in [-1.0, 1.0],
  "emotional_intensity": integer 1..5,
  "short_quote": string
}
Guidance:
- Pain: user expresses a problem/frustration without explicitly asking for help.
- Solution Seeking: user asks for help, advice, or recommendations.
- Workaround: user describes steps they took that others can reuse.
Pick the BEST FIT as the primary category; don't default to Pain.
Return JSON ONLY.
"""

def _probable_intent(title: str, text: str) -> str:
    t = f"{title}\n{text}".lower()
    if any(re.search(p, t) for p in WORK_PATTERNS): return "workaround"
    if "?" in t or any(re.search(p, t) for p in ASK_PATTERNS): return "ask"
    return ""

def _infer_flags(title: str, text: str, sent: float) -> Tuple[bool, bool, bool]:
    t = f"{title}\n{text}".lower()
    is_seek = ("?" in t) or any(k in t for k in GENERIC_QUESTION_TRIGGERS) or any(re.search(p, t) for p in ASK_PATTERNS)
    is_work = any(k in t for k in GENERIC_WORKAROUND_TRIGGERS) or any(re.search(p, t) for p in WORK_PATTERNS)
    is_pain = (sent < -0.15) or any(k in t for k in GENERIC_PAIN_TRIGGERS)
    return is_pain, is_seek, is_work

def gemini_label(model, problem: str, row: Dict) -> Dict:
    hint = _probable_intent(row["title"], row["text"])
    content = (
        f"Problem theme: {problem}\n"
        f"Subreddit: r/{row['subreddit']}\n"
        f"Title: {row['title']}\n"
        f"Body:\n{row['text']}\n"
        f"(Heuristic hint: {hint or 'none'})"
    )
    try:
        resp = model.generate_content([CLS_PROMPT, content])
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""
    m = re.search(r"\{.*\}", raw, flags=re.S)
    out = {
        "category": "Pain", "intent": "complaint",
        "sentiment_compound": -0.1, "emotional_intensity": 2,
        "short_quote": short(row["text"] or row["title"])
    }
    if m:
        try:
            obj = json.loads(m.group(0))
            cat = obj.get("category", "Pain")
            if cat not in ("Pain","Solution Seeking","Workaround"): cat = "Pain"
            intent = obj.get("intent", "complaint")
            sent = float(obj.get("sentiment_compound", -0.1))
            intensity = max(1, min(5, int(obj.get("emotional_intensity", 2))))
            quote = short(obj.get("short_quote") or row["text"] or row["title"])
            out = {"category": cat, "intent": intent,
                   "sentiment_compound": sent, "emotional_intensity": intensity,
                   "short_quote": quote}
        except Exception:
            pass
    if out["category"] == "Pain" and hint == "ask":
        out["category"] = "Solution Seeking"; out["intent"] = "ask"
    if out["category"] == "Pain" and hint == "workaround":
        out["category"] = "Workaround"; out["intent"] = "workaround"
    return out

# ---------- Scoring + Report ----------
def compute_frs_pvi(df: pd.DataFrame, F_cap=F_CAP_DEFAULT, H=RECENCY_HALF_LIFE, weights=WEIGHTS):
    N = max(len(df), 1)
    F = 100.0 * min(N / float(F_cap), 1.0)
    R_raw = df["days_ago"].apply(lambda d: math.exp(-d / H)).mean() if N else 0.0
    R = 100.0 * R_raw
    def mean(cat):
        v = df.loc[df["category"] == cat, "sentiment_compound"].mean()
        return 0.0 if pd.isna(v) else float(v)
    pain_m = mean("Pain")
    seek_m = mean("Solution Seeking")
    work_m = mean("Workaround")
    S_raw = clamp(- (0.5*pain_m + 0.35*seek_m + 0.15*work_m), 0.0, 1.0)
    S = 100.0 * S_raw
    wF, wR, wS = weights
    PVI = wF*F + wR*R + wS*S
    rating = "Strong" if PVI >= 70 else ("Moderate" if PVI >= 40 else "Weak")
    return {"N": int(N), "F": round(F,1), "R": round(R,1), "S": round(S,1), "PVI": round(PVI,1), "rating": rating}

def hr(): print("-" * 72)

def sentiment_cards(df: pd.DataFrame):
    total = len(df)
    neg = int((df["sentiment_compound"] < -0.15).sum())
    pos = int((df["sentiment_compound"] > 0.15).sum())
    neu = int(total - neg - pos)
    print("  Negative ", f"{neg:<3}", f"{pct(neg,total)}%")
    print("  Positive ", f"{pos:<3}", f"{pct(pos,total)}%")
    print("  Neutral  ", f"{neu:<3}", f"{pct(neu,total)}%")

def section(title, subdf, badge_label="NEGATIVE"):
    print(f"\n{title}"); hr()
    if subdf.empty:
        print("  (no signals in this run)"); return
    tmp = subdf.copy()

    # Prefer items that are actually on-topic for the problem
    if "relevance_score" in tmp.columns:
        # Filter out clearly off-topic stuff if possible
        filtered = tmp[tmp["relevance_score"] >= 0.08]
        if not filtered.empty:
            tmp = filtered

        tmp["rank_key"] = list(zip(
            -tmp["relevance_score"],
            -tmp["emotional_intensity"],
            tmp["days_ago"],
            tmp["sentiment_compound"]
        ))
    else:
        tmp["rank_key"] = list(zip(
            -tmp["emotional_intensity"],
            tmp["days_ago"],
            tmp["sentiment_compound"]
        ))

    tmp = tmp.sort_values("rank_key").head(4)
    for _, r in tmp.iterrows():
        badge = (int(round(100 * clamp(-r["sentiment_compound"], 0, 1)))
                 if badge_label == "NEGATIVE"
                 else min(int(round(50 * clamp(r["sentiment_compound"] + 0.2, 0, 1)
                                   + 10 * r["emotional_intensity"])), 100))
        print(f'  "{r["short_quote"]}"')
        print(f"   — r/{r['subreddit']} · {utc_str(r['created_utc'])} · {badge_label} {badge}%")
        print(f"     {r['permalink']}")

def render_console_report(problem: str, df: pd.DataFrame, pvi: Dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        console_report(problem, df, pvi)
    return buf.getvalue()

def console_report(problem: str, df: pd.DataFrame, pvi: Dict):
    total = len(df)
    pain_overlap = int(df["is_pain"].sum())
    seek_overlap = int(df["is_seek"].sum())
    work_overlap = int(df["is_work"].sum())

    seek_sig = df["is_seek"] & (df["emotional_intensity"] >= 2)
    work_sig = df["is_work"] & (df["sentiment_compound"] > -0.3)
    opp_overlap = int((seek_sig | work_sig).sum())

    oldest = utc_str(df["created_utc"].min())
    newest = utc_str(df["created_utc"].max())

    print("\nM2  •  Problem Validation Brief"); hr()
    print(f"Assessment: {pvi['rating']}  •  {pct(pain_overlap,total)}% pain  •  {pct(opp_overlap,total)}% opportunity (any seek/work)\n")
    print(
        f"Out of {total} posts analyzed (buckets can overlap), "
        f"{pain_overlap} ({pct(pain_overlap,total)}%) showed pain, "
        f"{seek_overlap} ({pct(seek_overlap,total)}%) were seeking solutions, and "
        f"{work_overlap} ({pct(work_overlap,total)}%) described workarounds."
    )
    print(f"Evidence window: {oldest} → {newest}.\n")
    sentiment_cards(df)
    section('Pain Signals', df[df['is_pain']], 'NEGATIVE')
    section('Solution-seeking', df[df['is_seek']], 'OPPORTUNITY')
    section('Workarounds', df[df['is_work']], 'OPPORTUNITY')

    print("\nScoring"); hr()
    print(f"Frequency (cap {F_CAP_DEFAULT}/window): {pvi['F']}/100")
    print(f"Recency (half-life {RECENCY_HALF_LIFE}d):  {pvi['R']}/100")
    print(f"Sentiment (neg → pain):                  {pvi['S']}/100")
    print(f"Final Problem Validation Index (PVI):    {pvi['PVI']}/100  →  {pvi['rating']}\n")

# ---------- NEW: LLM-based query builder ----------
def llm_build_queries(problem: str, max_queries: int = 6) -> List[str]:
    try:
        model = init_gemini()
    except SystemExit:
        return []
    except Exception:
        return []

    prompt = f"""
You are helping with problem validation for startup ideas.

Input is a detailed problem statement a founder wrote.
Your job is to turn it into a few short Google search queries
that will likely surface Reddit threads where real people:
- complain about this problem (Pain),
- ask for help or recommendations (Solution Seeking), or
- describe how they solved it (Workarounds).

Rules:
- Return STRICT JSON: {{"queries": ["q1", "q2", "..."]}}
- 3 to {max_queries} queries.
- Each query should be 12 words or fewer.
- Prefer everyday language people would use when venting or asking on Reddit.
- You may include terms like "reddit", "data scientist", "ml engineer", "operations", etc.
- Avoid overlong or highly academic phrasing.
"""

    content = f"Problem statement:\n{problem}"
    try:
        resp = model.generate_content([prompt, content])
        raw = (resp.text or "").strip()
    except Exception:
        return []

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        queries = obj.get("queries") or []
        clean = []
        for q in queries:
            if isinstance(q, str):
                q = q.strip()
                if q:
                    clean.append(q)
        return clean
    except Exception:
        return []

# ---------- UPDATED build_queries ----------
def build_queries(problem: str) -> List[str]:
    problem = (problem or "").strip()
    if not problem:
        return []

    if len(problem.split()) <= 12:
        base = problem.strip()
        seeds = [
            f'{base} site:reddit.com',
            f'"{base}" site:reddit.com',
            f'how to {base} site:reddit.com',
            f'{base} not working site:reddit.com',
            f'{base} workaround site:reddit.com',
            f'{base} recommendations site:reddit.com',
        ]
    else:
        llm_q = llm_build_queries(problem)
        if llm_q:
            seeds = []
            for q in llm_q:
                if "reddit" in q.lower():
                    seeds.append(q)
                else:
                    seeds.append(f"{q} reddit")
        else:
            core_phrase, keyword_phrase = _condense_problem(problem)
            bases = []
            if core_phrase:
                bases.append(core_phrase)
            if keyword_phrase and keyword_phrase != core_phrase.lower():
                bases.append(keyword_phrase)

            seeds = []
            for b in bases:
                seeds.append(f'{b} site:reddit.com')
                seeds.append(f'"{b}" site:reddit.com')
                seeds.append(f'how to {b} site:reddit.com')
                seeds.append(f'{b} not working site:reddit.com')
                seeds.append(f'{b} struggling site:reddit.com')
                seeds.append(f'{b} advice site:reddit.com')
                seeds.append(f'{b} workaround site:reddit.com')
                seeds.append(f'{b} reddit')

    out, seen = [], set()
    for q in seeds:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

# ---------- NEW: relevance scoring ----------
def _relevance_scores(df: pd.DataFrame, problem: str, queries: List[str]) -> List[float]:
    """
    Cheap lexical relevance score in [0,1] per row, based on overlap with:
    - problem tokens
    - tokens of the queries we used to search Reddit
    """
    base_tokens = set(_tok(problem))
    query_token_sets = [set(_tok(q)) for q in (queries or []) if q]

    scores: List[float] = []
    for _, row in df.iterrows():
        text_tokens = set(_tok((row.get("title") or "") + " " + (row.get("text") or "")))
        if not text_tokens:
            scores.append(0.0)
            continue

        max_q = 0.0
        for qt in query_token_sets:
            if not qt:
                continue
            inter = len(text_tokens & qt)
            max_q = max(max_q, inter / max(1, len(qt)))

        if base_tokens:
            inter_p = len(text_tokens & base_tokens)
            prob_score = inter_p / max(1, len(base_tokens))
        else:
            prob_score = 0.0

        score = max(max_q, prob_score)
        scores.append(float(score))

    return scores

# ----------------- service helpers -----------------
import requests

def serpapi_search(query: str, max_results: int = 10):
    """Search Google for Reddit threads using SerpAPI."""
    if not SERPAPI_KEY:
        print("Warning: SERPAPI_KEY not set")
        return []
    
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return [r.get("link") for r in data.get("organic_results", []) if r.get("link")]
        else:
            print(f"SerpAPI error: {resp.status_code} {resp.text}")
            return []
    except Exception as e:
        print(f"SerpAPI exception: {e}")
        return []

def normalize_reddit_url(url: str) -> str:
    """Ensure URL is clean for JSON fetching."""
    if "?" in url:
        url = url.split("?")[0]
    if not url.endswith("/"):
        url += "/"
    return url

def fetch_reddit_json(url: str):
    """Fetch JSON data from a Reddit URL."""
    jurl = url if url.endswith(".json") else url + ".json"
    try:
        # User-agent is required by Reddit
        headers = {"User-Agent": "MarketOS-M2/1.0"}
        resp = requests.get(jurl, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def extract_post(payload):
    """Extract main post data from Reddit JSON payload."""
    try:
        if isinstance(payload, list) and len(payload) > 0:
            data = payload[0].get("data", {}).get("children", [{}])[0].get("data", {})
            return {
                "kind": "post",
                "subreddit": data.get("subreddit"),
                "title": data.get("title"),
                "text": data.get("selftext", ""),
                "author": data.get("author"),
                "num_comments": data.get("num_comments", 0),
                "score": data.get("score", 0),
                "permalink": data.get("permalink"),
                "created_utc": data.get("created_utc", 0)
            }
    except Exception:
        pass
    return None

def flatten_comments(children, depth=0):
    """Recursively flatten Reddit comment tree."""
    flat = []
    for c in children:
        if c.get("kind") == "t1":
            data = c.get("data", {})
            flat.append(data)
            if data.get("replies"):
                replies = data["replies"].get("data", {}).get("children", [])
                flat.extend(flatten_comments(replies, depth + 1))
    return flat

def ascii_alpha_ratio(text: str) -> float:
    """Calculate ratio of ASCII alphabetic characters."""
    if not text: return 0.0
    alphas = sum(1 for c in text if c.isalpha() and c.isascii())
    return alphas / len(text)

def jaccard_3gram(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity of 3-grams."""
    if not s1 or not s2: return 0.0
    def get_grams(s):
        return set(s[i:i+3] for i in range(len(s)-2))
    g1, g2 = get_grams(s1), get_grams(s2)
    if not g1 or not g2: return 0.0
    return len(g1 & g2) / len(g1 | g2)

def heuristic_sentiment(title: str, text: str) -> float:
    """Simple sentiment heuristic based on keywords."""
    content = (title + " " + text).lower()
    score = 0.0
    # Very basic counts
    for w in SENT_POS:
        if w in content: score += 0.2
    for w in SENT_NEG:
        if w in content: score -= 0.3
    return max(-1.0, min(1.0, score))

def _tok(s: str):
    """Simple tokenizer."""
    return re.findall(r"\w+", s.lower())

def _infer_flags(title: str, text: str, sent: float):
    """Infer flags (Pain, Seek, Work) based on content and sentiment."""
    content = (title + " " + text).lower()
    is_pain = sent < -0.2 or any(w in content for w in GENERIC_PAIN_TRIGGERS)
    is_seek = any(w in content for w in GENERIC_QUESTION_TRIGGERS)
    is_work = any(w in content for w in GENERIC_WORKAROUND_TRIGGERS)
    return is_pain, is_seek, is_work



def gemini_label(model, problem, row):
    """Label a row using Gemini/OpenAI."""
    # Placeholder or simple implementation if needed, 
    # but _analyze calls it.
    # If model is None, return dummy.
    if not model:
        return {"category": "Pain", "intent": "complaint", "sentiment_compound": -0.5}
    
    # Construct prompt
    prompt = f"""
    Problem: {problem}
    Post Title: {row.get('title')}
    Post Text: {row.get('text')[:500]}
    
    Classify this post.
    """
    try:
        resp = model.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = resp.choices[0].message.content
        # Clean markdown
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Label error: {e}")
        return {"category": "Pain", "intent": "complaint", "sentiment_compound": 0.0}


def _analyze(problem: str, max_results: int):
    queries = build_queries(problem)
    urls: List[str] = []
    for q in queries:
        if len(urls) >= max_results: break
        hits = serpapi_search(q, max_results=max(1, max_results - len(urls)))
        urls.extend(hits)

    if not urls:
        return pd.DataFrame(), {"N": 0, "F": 0.0, "R": 0.0, "S": 0.0, "PVI": 0.0, "rating": "Weak"}

    urls = [normalize_reddit_url(u) for u in urls]
    dedup, seen = [], set()
    for u in urls:
        key_m = re.search(r"/comments/([a-z0-9]{6,8})/", u)
        key = key_m.group(1) if key_m else u
        if key not in seen:
            seen.add(key); dedup.append(u)
        if len(dedup) >= max_results: break

    rows: List[Dict[str, Any]] = []
    for u in tqdm(dedup, ncols=80, desc="Fetching"):
        payload = fetch_reddit_json(u)
        if not payload:
            time.sleep(0.2); continue

        post = extract_post(payload)
        if post:
            rows.append(post)
            try:
                comments_listing = payload[1]
                children = comments_listing.get("data", {}).get("children", [])
                flat = flatten_comments(children, depth=0)

                kept = 0
                dedup_buf: List[str] = []
                for c in flat:
                    if kept >= COMMENTS_PER_THREAD: break
                    body = c["body"]
                    if len((body or "").split()) < MIN_COMMENT_WORDS: continue
                    if c.get("score", 0) < MIN_COMMENT_SCORE: continue
                    if ascii_alpha_ratio(body) < 0.6: continue
                    if any(jaccard_3gram(body, prev) > 0.90 for prev in dedup_buf): continue

                    rows.append({
                        "kind": "comment",
                        "subreddit": post["subreddit"],
                        "title": post["title"],
                        "text": body,
                        "author": c["author"],
                        "num_comments": post["num_comments"],
                        "score": c["score"],
                        "permalink": c["permalink"] or post["permalink"],
                        "created_utc": c["created_utc"]
                    })
                    dedup_buf.append(body); kept += 1
            except Exception:
                pass

        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(), {"N": 0, "F": 0.0, "R": 0.0, "S": 0.0, "PVI": 0.0, "rating": "Weak"}

    df = pd.DataFrame(rows).drop_duplicates(subset=["permalink","text"]).reset_index(drop=True)
    df["days_ago"] = df["created_utc"].apply(days_ago)
    df = df[df["days_ago"] <= MAX_DAYS_OLD].reset_index(drop=True)
    if df.empty:
        return df, {"N": 0, "F": 0.0, "R": 0.0, "S": 0.0, "PVI": 0.0, "rating": "Weak"}

    df["thread_id"] = df["permalink"].str.extract(r"/comments/([a-z0-9]{6,8})/")
    df = df.sort_values(["thread_id","score"], ascending=[True, False])
    df["thread_rank"] = df.groupby("thread_id").cumcount()
    def thread_weight(rank): return 1.0 if rank == 0 else 0.8 if rank == 1 else 0.6 if rank == 2 else 0.5
    df["weight"] = df["thread_rank"].apply(thread_weight)

    df["sentiment_compound"] = [heuristic_sentiment(ti, tx) for ti, tx in zip(df["title"], df["text"])]

    def infer_flags_row(r):
        return _infer_flags(r["title"], r["text"], r["sentiment_compound"])
    flags = [infer_flags_row(r) for _, r in df.iterrows()]
    df["is_pain"], df["is_seek"], df["is_work"] = zip(*flags)

    def primary_cat(row):
        if row["is_seek"]: return "Solution Seeking"
        if row["is_work"]: return "Workaround"
        if row["is_pain"]: return "Pain"
        return "Pain"
    df["category"] = df.apply(primary_cat, axis=1)
    df["intent"] = df["category"].map({"Pain":"complaint","Solution Seeking":"ask","Workaround":"workaround"})
    df["emotional_intensity"] = [2 + int(3 * abs(s)) for s in df["sentiment_compound"]]
    df["short_quote"] = [short(x or y) for x, y in zip(df["text"], df["title"])]

    # NEW: relevance scores (used only for what we show in report/highlights)
    df["relevance_score"] = _relevance_scores(df, problem, queries)

    def uncertainty(row):
        amb = int((row["is_seek"] and row["is_work"]) or (not row["is_seek"] and not row["is_work"] and not row["is_pain"]))
        sent_flat = 1.0 - abs(row["sentiment_compound"])
        length_pen = 1.0 if len((row["text"] or "").split()) < 25 else 0.0
        recency_boost = math.exp(-row["days_ago"]/RECENCY_HALF_LIFE)
        thread_boost = 0.05 if row.get("thread_rank", 0) <= 2 else 0.0
        return 0.6*amb + 0.3*sent_flat + 0.1*length_pen + 0.2*recency_boost + thread_boost

    df["uncertainty"] = df.apply(uncertainty, axis=1)
    k = min(MAX_LLM, len(df))
    df_llm = df.sort_values("uncertainty", ascending=False).head(k).copy()

    model = init_gemini()
    for i, r in df_llm.iterrows():
        try:
            res = gemini_label(model, problem, dict(r))
            llm_sent = float(res.get("sentiment_compound", 0.0))
            if abs(llm_sent) >= 0.2:
                df.at[i, "sentiment_compound"] = llm_sent
            df.at[i, "category"] = res["category"]
            df.at[i, "intent"] = res["intent"]
            df.at[i, "emotional_intensity"] = int(res["emotional_intensity"])
            df.at[i, "short_quote"] = res["short_quote"]
            p, s, w = _infer_flags(r["title"], r["text"], df.at[i,"sentiment_compound"])
            df.at[i,"is_pain"], df.at[i,"is_seek"], df.at[i,"is_work"] = p, s, w
        except Exception:
            pass
        if i % 5 == 0:
            time.sleep(0.05)

    pvi = compute_frs_pvi(df)
    return df, pvi

def _top_cards(subdf: pd.DataFrame, badge_label: str):
    out = []
    if subdf.empty: return out
    tmp = subdf.copy()

    if "relevance_score" in tmp.columns:
        filtered = tmp[tmp["relevance_score"] >= 0.08]
        if not filtered.empty:
            tmp = filtered
        tmp["rank_key"] = list(zip(
            -tmp["relevance_score"],
            -tmp["emotional_intensity"],
            tmp["days_ago"],
            tmp["sentiment_compound"]
        ))
    else:
        tmp["rank_key"] = list(zip(
            -tmp["emotional_intensity"],
            tmp["days_ago"],
            tmp["sentiment_compound"]
        ))

    tmp = tmp.sort_values("rank_key").head(4)
    for _, r in tmp.iterrows():
        if badge_label == "NEGATIVE":
            badge = int(round(100 * clamp(-r["sentiment_compound"], 0, 1)))
        else:
            badge = min(int(round(50 * clamp(r["sentiment_compound"] + 0.2, 0, 1)
                                   + 10 * r["emotional_intensity"])), 100)
        out.append({
            "quote": r["short_quote"],
            "subreddit": r["subreddit"],
            "created_utc": float(r["created_utc"]),
            "permalink": r["permalink"],
            "badge": badge,
            "badge_label": badge_label
        })
    return out



    total = int(len(df))
    pain_overlap = int(df["is_pain"].sum())
    seek_overlap = int(df["is_seek"].sum())
    work_overlap = int(df["is_work"].sum())

    seek_sig = df["is_seek"] & (df["emotional_intensity"] >= 2)
    work_sig = df["is_work"] & (df["sentiment_compound"] > -0.3)
    opp_overlap = int((seek_sig | work_sig).sum())

    oldest = utc_str(df["created_utc"].min())
    newest = utc_str(df["created_utc"].max())

    out = {
        "summary": {
            "problem": prompt,
            "total": total,
            "pain": pain_overlap,
            "seek": seek_overlap,
            "work": work_overlap,
            "pain_pct": pct(pain_overlap, total),
            "opportunity_pct": pct(opp_overlap, total)
        },
        "pvi": pvi,
        "window": {"oldest": oldest, "newest": newest},
        "highlights": {
            "pain": _top_cards(df[df["is_pain"]], "NEGATIVE"),
            "seek": _top_cards(df[df["is_seek"]], "OPPORTUNITY"),
            "work": _top_cards(df[df["is_work"]], "OPPORTUNITY")
        }
    }

    if include_report:
        out["report_text"] = render_console_report(prompt, df, pvi)

    return out

# ----------------- CLI main -----------------
def main():
    ap = argparse.ArgumentParser(description="StudioOS M2 – SerpApi Reddit (.json), minimal flags (polished)")
    ap.add_argument("--problem", required=True, help="Problem/theme to validate")
    ap.add_argument("--max-results", type=int, default=10, help="Total reddit threads to fetch")
    args = ap.parse_args()

    queries = build_queries(args.problem)
    urls: List[str] = []
    for q in queries:
        if len(urls) >= args.max_results: break
        hits = serpapi_search(q, max_results=max(1, args.max_results - len(urls)))
        urls.extend(hits)

    if not urls:
        print("No Reddit results found. Try rephrasing the problem.")
        return

    urls = [normalize_reddit_url(u) for u in urls]
    dedup, seen = [], set()
    for u in urls:
        key_m = re.search(r"/comments/([a-z0-9]{6,8})/", u)
        key = key_m.group(1) if key_m else u
        if key not in seen:
            seen.add(key); dedup.append(u)
        if len(dedup) >= args.max_results: break

    rows: List[Dict[str, Any]] = []
    from tqdm import tqdm as _tqdm
    for u in _tqdm(dedup, ncols=80, desc="Fetching"):
        payload = fetch_reddit_json(u)
        if not payload:
            time.sleep(0.2); continue

        post = extract_post(payload)
        if post:
            rows.append(post)
            try:
                comments_listing = payload[1]
                children = comments_listing.get("data", {}).get("children", [])
                flat = flatten_comments(children, depth=0)

                kept = 0
                dedup_buf: List[str] = []
                for c in flat:
                    if kept >= COMMENTS_PER_THREAD: break
                    body = c["body"]
                    if len((body or "").split()) < MIN_COMMENT_WORDS:
                        continue
                    if c.get("score", 0) < MIN_COMMENT_SCORE:
                        continue
                    if ascii_alpha_ratio(body) < 0.6:
                        continue
                    if any(jaccard_3gram(body, prev) > 0.90 for prev in dedup_buf):
                        continue

                    rows.append({
                        "kind": "comment",
                        "subreddit": post["subreddit"],
                        "title": post["title"],
                        "text": body,
                        "author": c["author"],
                        "num_comments": post["num_comments"],
                        "score": c["score"],
                        "permalink": c["permalink"] or post["permalink"],
                        "created_utc": c["created_utc"]
                    })
                    dedup_buf.append(body)
                    kept += 1
            except Exception:
                pass

        time.sleep(0.2)

    if not rows:
        print("No quality content after filtering (score/length). Loosen thresholds or try a new query.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["permalink","text"]).reset_index(drop=True)
    df["days_ago"] = df["created_utc"].apply(days_ago)
    df = df[df["days_ago"] <= MAX_DAYS_OLD].reset_index(drop=True)
    if df.empty:
        print("Everything was older than MAX_DAYS_OLD; try a broader problem or raise the cap.")
        return

    df["thread_id"] = df["permalink"].str.extract(r"/comments/([a-z0-9]{6,8})/")
    df = df.sort_values(["thread_id","score"], ascending=[True, False])
    df["thread_rank"] = df.groupby("thread_id").cumcount()
    def thread_weight(rank): return 1.0 if rank == 0 else 0.8 if rank == 1 else 0.6 if rank == 2 else 0.5
    df["weight"] = df["thread_rank"].apply(thread_weight)

    df["sentiment_compound"] = [heuristic_sentiment(ti, tx) for ti, tx in zip(df["title"], df["text"])]

    def infer_flags_row(r):
        return _infer_flags(r["title"], r["text"], r["sentiment_compound"])
    flags = [infer_flags_row(r) for _, r in df.iterrows()]
    df["is_pain"], df["is_seek"], df["is_work"] = zip(*flags)

    def primary_cat(row):
        if row["is_seek"]: return "Solution Seeking"
        if row["is_work"]: return "Workaround"
        if row["is_pain"]: return "Pain"
        return "Pain"
    df["category"] = df.apply(primary_cat, axis=1)
    df["intent"] = df["category"].map({"Pain":"complaint","Solution Seeking":"ask","Workaround":"workaround"})
    df["emotional_intensity"] = [2 + int(3 * abs(s)) for s in df["sentiment_compound"]]
    df["short_quote"] = [short(x or y) for x, y in zip(df["text"], df["title"])]

    # NEW: relevance scores for CLI report as well
    df["relevance_score"] = _relevance_scores(df, args.problem, queries)

    def uncertainty(row):
        amb = int((row["is_seek"] and row["is_work"]) or (not row["is_seek"] and not row["is_work"] and not row["is_pain"]))
        sent_flat = 1.0 - abs(row["sentiment_compound"])
        length_pen = 1.0 if len((row["text"] or "").split()) < 25 else 0.0
        recency_boost = math.exp(-row["days_ago"]/RECENCY_HALF_LIFE)
        thread_boost = 0.05 if row.get("thread_rank", 0) <= 2 else 0.0
        return 0.6*amb + 0.3*sent_flat + 0.1*length_pen + 0.2*recency_boost + thread_boost

    df["uncertainty"] = df.apply(uncertainty, axis=1)
    k = min(MAX_LLM, len(df))
    df_llm = df.sort_values("uncertainty", ascending=False).head(k).copy()

    model = init_gemini()
    for i, r in df_llm.iterrows():
        try:
            res = gemini_label(model, args.problem, dict(r))
            llm_sent = float(res.get("sentiment_compound", 0.0))
            if abs(llm_sent) >= 0.2:
                df.at[i, "sentiment_compound"] = llm_sent
            df.at[i, "category"] = res["category"]
            df.at[i, "intent"] = res["intent"]
            df.at[i, "emotional_intensity"] = int(res["emotional_intensity"])
            df.at[i, "short_quote"] = res["short_quote"]
            p, s, w = _infer_flags(r["title"], r["text"], df.at[i,"sentiment_compound"])
            df.at[i,"is_pain"], df.at[i,"is_seek"], df.at[i,"is_work"] = p, s, w
        except Exception:
            pass
        if i % 5 == 0:
            time.sleep(0.05)

    pvi = compute_frs_pvi(df)
    console_report(args.problem, df, pvi)

if __name__ == "__main__":
    main()

# =========================
# Adapter for StudioOS Flask Backend
# =========================

def run_m2(problem: str, api_key: str = None, max_results: int = 10, include_report: bool = False) -> dict:
    """
    Main function for M2 module (StudioOS Adapter).
    Executes the problem validation using the latest logic.
    """
    try:
        # Configure OpenAI client with dynamic key
        global OPENAI_API_KEY
        if api_key:
            OPENAI_API_KEY = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Run the analysis
        df, pvi = _analyze(problem, max_results=max_results)
        
        # Convert DataFrame to list of dicts for JSON response
        posts_data = []
        if not df.empty:
            # Handle potential NaN values
            df = df.fillna("")
            posts_data = df.to_dict(orient="records")
            
        # Construct summary stats
        summary = {
            "total_posts": int(pvi.get("N", 0)),
            "pain_signals": int(df["is_pain"].sum()) if not df.empty else 0,
            "seek_signals": int(df["is_seek"].sum()) if not df.empty else 0,
            "work_signals": int(df["is_work"].sum()) if not df.empty else 0,
        }
        
        # # Build citations list
        # citations = []
        # if not df.empty:
        #     for _, row in df.head(20).iterrows():
        #         citations.append({
        #             "title": row.get("title", "Reddit Post"),
        #             "uri": row.get("permalink", ""),
        #             "source": "Reddit"
        #         })

        # Build citations list
        citations = []
        if not df.empty:
            for _, row in df.head(20).iterrows():
                raw = (row.get("permalink") or "").strip()

                # If Reddit returned a relative permalink like "/r/…",
                # turn it into a full URL so the frontend opens reddit.com,
                # not the StudioOS origin.
                if raw and not raw.startswith("http"):
                    raw = f"https://www.reddit.com{raw}"

                citations.append({
                    "title": row.get("title", "Reddit Post"),
                    "uri": raw,
                    "source": "Reddit"
                })


        return {
            "assessment": pvi.get("rating", "Weak"),
            "pvi": pvi,
            "summary": summary,
            "posts": posts_data[:20], # Limit to top 20
            "citations": citations,   # Standardized citations
            "problem": problem,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        # logging.error(f"Error in run_m2: {e}", exc_info=True)
        return {
            "assessment": "Error",
            "pvi": {"rating": "Error", "PVI": 0},
            "summary": {},
            "posts": [],
            "error": str(e)
        }
