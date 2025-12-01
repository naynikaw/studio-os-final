"""
Module M2: Problem Validation
Refactored for FastAPI integration
"""
import re
import math
import time
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# Suppress gRPC/ALTS warnings
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")

import pandas as pd
import praw
import google.generativeai as genai

# =============== API KEYS ===============
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "yr3ojj7Dfu0MW2FF9SWCrQ")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "K8ZnT8yaU-idVU86Ec660HwMkUVBTw")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "Module 2 MVP validation script for CMU project by u/Neighbourhood_Junkie")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA0HVszwZjrylWwUITh7lbD0rBbwzf1I-Q")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# ---------------- Defaults ----------------
DEFAULT_SUBS = ["AskReddit", "explainlikeimfive"]
F_CAP_DEFAULT = 50
RECENCY_HALF_LIFE = 30.0
WEIGHTS = (0.40, 0.35, 0.25)
TOTAL_HARD_CAP = 100

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

# --------------- Reddit ---------------------
def connect_reddit():
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        raise ValueError("REDDIT credentials not configured")
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        ratelimit_seconds=60
    )

def build_queries(problem: str, mode: str) -> List[str]:
    base = problem.strip()
    syn = ["tips","advice","tools","apps","workflow","workaround","alternative",
           "not working","fix","best way","guide","help","problem","error"]
    seeds = [
        f'"{base}"',
        f'how to {base}',
        f'how do i {base}',
        f'{base} recommendations',
        f'{base} advice',
        f'{base} not working',
        f'fix {base}',
        f'{base} workaround',
        f'{base} guide',
    ]
    expanded = []
    for s in seeds:
        expanded.append(s)
        if mode == "thorough":
            for k in syn:
                expanded.append(f'{s} "{k}"')
    if mode == "fast":
        expanded = expanded[:6]
    out, seen = [], set()
    for q in expanded:
        if q not in seen:
            seen.add(q); out.append(q)
    return out

def _terms_from(text: str):
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in raw if len(t) > 2]

def scrape_reddit(problem: str, subreddits: List[str], time_filter: str,
                  mode: str, per_query_limit: int, max_per_sub: int) -> List[Dict]:
    reddit = connect_reddit()
    queries = build_queries(problem, mode)
    base_terms = set(_terms_from(problem))

    rows, seen_links = [], set()
    kept, skipped_promo, skipped_irrelevant = 0, 0, 0

    strategies = [("new", False)] if mode == "fast" else [
        ("relevance", True), ("top", True), ("relevance", False), ("top", False), ("new", False)
    ]

    banned_phrases = [
        "newsletter","recap","roundup","this week","weekly update","subscribe","follow my",
        "launching soon","promo code","discount","hiring","job opening","internship",
        "giveaway","press release","funding round","pitch deck"
    ]

    per_sub_count = {s: 0 for s in subreddits}

    for sort_mode, require_all in strategies:
        if len(rows) >= TOTAL_HARD_CAP: break

        for sub in subreddits:
            if len(rows) >= TOTAL_HARD_CAP: break
            if per_sub_count[sub] >= max_per_sub: continue

            sr = reddit.subreddit(sub)
            for q in queries:
                if len(rows) >= TOTAL_HARD_CAP: break
                if per_sub_count[sub] >= max_per_sub: break

                try:
                    for p in sr.search(q, sort=sort_mode, time_filter=time_filter, limit=per_query_limit):
                        if len(rows) >= TOTAL_HARD_CAP: break
                        if per_sub_count[sub] >= max_per_sub: break

                        title = p.title or ""
                        text = (p.selftext or "").strip()
                        if not title and not text:
                            continue

                        hay = (title + " " + text).lower()

                        if any(bp in hay for bp in banned_phrases):
                            skipped_promo += 1; continue
                        if len(text.split()) < 10 and "http" in text:
                            skipped_promo += 1; continue

                        if base_terms:
                            if require_all:
                                taken = all(t in hay for t in base_terms)
                            else:
                                hits = sum(t in hay for t in base_terms)
                                taken = (hits >= 2) or (problem.lower() in hay)
                        else:
                            taken = True
                        if not taken:
                            skipped_irrelevant += 1; continue

                        link = f"https://www.reddit.com{p.permalink}"
                        if link in seen_links:
                            continue
                        seen_links.add(link)

                        rows.append({
                            "source": "reddit",
                            "subreddit": p.subreddit.display_name,
                            "query": q,
                            "title": title,
                            "text": text,
                            "author": str(p.author) if p.author else "[deleted]",
                            "num_comments": getattr(p, "num_comments", 0),
                            "score": getattr(p, "score", 0),
                            "permalink": link,
                            "created_utc": float(p.created_utc),
                        })
                        kept += 1
                        per_sub_count[sub] += 1

                    time.sleep(0.1 if mode == "fast" else 0.2)
                except Exception as e:
                    print(f"[warn] r/{sub} ({sort_mode}) '{q}': {e}")
                    time.sleep(0.4 if mode == "fast" else 0.6)

        if mode == "fast" and kept >= min(TOTAL_HARD_CAP, 40):
            break

    return rows[:TOTAL_HARD_CAP]

# --------------- Gemini ---------------------
def init_gemini():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

def _probable_intent_from_text(title: str, text: str, qtrigs, wtrigs) -> str:
    t = f"{title}\n{text}".lower()
    if any(k in t for k in wtrigs): return "workaround"
    if "?" in t or any(k in t for k in qtrigs): return "ask"
    return ""

def _infer_flags(title: str, text: str, sentiment_compound: float,
                 qtrigs, wtrigs, ptrigs) -> Tuple[bool, bool, bool]:
    t = f"{title}\n{text}".lower()
    is_seek = ("?" in t) or any(k in t for k in qtrigs)
    is_work = any(k in t for k in wtrigs)
    is_pain = (sentiment_compound < -0.15) or any(k in t for k in ptrigs)
    return is_pain, is_seek, is_work

def _tok(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def heuristic_sentiment(title: str, text: str) -> float:
    toks = _tok(title) + _tok(text)
    pos = sum(any(p in t for p in SENT_POS) for t in toks)
    neg = sum(any(n in t for n in SENT_NEG) for t in toks) * 1.2
    denom = pos + neg + 5.0
    score = (pos - neg) / denom
    return float(max(-1.0, min(1.0, score)))

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

def gemini_label(model, problem: str, title: str, text: str, subreddit: str, qtrigs, wtrigs) -> Dict:
    hint = _probable_intent_from_text(title, text, qtrigs, wtrigs)
    content = (
        f"Problem theme: {problem}\n"
        f"Subreddit: r/{subreddit}\n"
        f"Title: {title}\n"
        f"Body:\n{text}\n"
        f"(Heuristic hint: {hint or 'none'})"
    )
    try:
        resp = model.generate_content([CLS_PROMPT, content])
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""

    m = re.search(r"\{.*\}", raw, flags=re.S)
    out = {
        "category": "Pain",
        "intent": "complaint",
        "sentiment_compound": -0.1,
        "emotional_intensity": 2,
        "short_quote": short(text or title)
    }

    if m:
        try:
            obj = json.loads(m.group(0))
            cat = obj.get("category", "Pain")
            if cat not in ("Pain","Solution Seeking","Workaround"): cat = "Pain"
            intent = obj.get("intent", "complaint")
            sent = float(obj.get("sentiment_compound", -0.1))
            intensity = max(1, min(5, int(obj.get("emotional_intensity", 2))))
            quote = short(obj.get("short_quote") or text or title)
            out = {
                "category": cat, "intent": intent,
                "sentiment_compound": sent, "emotional_intensity": intensity,
                "short_quote": quote
            }
        except Exception:
            pass

    if out["category"] == "Pain" and hint == "ask":
        out["category"] = "Solution Seeking"; out["intent"] = "ask"
    if out["category"] == "Pain" and hint == "workaround":
        out["category"] = "Workaround"; out["intent"] = "workaround"
    return out

# ----------- FRS PVI Scoring ---------------
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

def exclusive_counts(df: pd.DataFrame):
    pain_only = df["is_pain"] & ~df["is_seek"] & ~df["is_work"]
    seek_only = df["is_seek"] & ~df["is_pain"] & ~df["is_work"]
    work_only = df["is_work"] & ~df["is_pain"] & ~df["is_seek"]
    return int(pain_only.sum()), int(seek_only.sum()), int(work_only.sum())


def run_m2(problem: str, mode: str = "fast", max_llm: int = 24,
           per_query_limit: int = 8, max_per_sub: int = 25) -> Dict:
    """
    Main function for M2 module.
    Returns JSON-serializable dict with validation results.
    """
    # Default subreddits
    subs = [
        "recruitinghell", "jobs", "career_advice", "cscareerquestions", "humanresources",
        "ArtificialIntelligence", "MachineLearning", "DataScience", "ChatGPT", "OpenAI",
        "Entrepreneur", "startups", "SaaS", "SideProject", "smallbusiness",
        "Productivity", "remotework", "workreform", "GetDisciplined",
        "technology", "Futurology", "learnprogramming",
        "AskReddit", "explainlikeimfive"
    ]
    
    time_filter = "month"
    
    # Scrape Reddit
    rows = scrape_reddit(problem, subs, time_filter, mode, per_query_limit, max_per_sub)
    if not rows:
        return {
            "assessment": "Weak",
            "message": "No posts found. Try a broader phrasing, more subs, or a longer time window.",
            "pvi": {"N": 0, "F": 0, "R": 0, "S": 0, "PVI": 0, "rating": "Weak"},
            "posts": []
        }
    
    df = pd.DataFrame(rows).drop_duplicates(subset=["permalink"]).reset_index(drop=True)
    df["days_ago"] = df["created_utc"].apply(days_ago)
    
    # PASS 1: heuristic labeling
    df["sentiment_compound"] = [heuristic_sentiment(ti, tx) for ti, tx in zip(df["title"], df["text"])]
    flags = [
        _infer_flags(r["title"], r["text"], r["sentiment_compound"],
                     GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS, GENERIC_PAIN_TRIGGERS)
        for _, r in df.iterrows()
    ]
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
    
    # Uncertainty scoring
    def uncertainty(row):
        amb = int((row["is_seek"] and row["is_work"]) or (not row["is_seek"] and not row["is_work"] and not row["is_pain"]))
        sent_flat = 1.0 - abs(row["sentiment_compound"])
        length_pen = 1.0 if len((row["text"] or "").split()) < 25 else 0.0
        recency_boost = math.exp(-row["days_ago"]/RECENCY_HALF_LIFE)
        return 0.6*amb + 0.3*sent_flat + 0.1*length_pen + 0.2*recency_boost
    
    df["uncertainty"] = df.apply(uncertainty, axis=1)
    df_llm_candidates = df.sort_values("uncertainty", ascending=False).head(int(max_llm)).copy()
    
    # PASS 2: LLM labeling
    model = init_gemini()
    for i, r in df_llm_candidates.iterrows():
        try:
            res = gemini_label(model, problem, r["title"], r["text"], r["subreddit"],
                               GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS)
            llm_sent = float(res.get("sentiment_compound", 0.0))
            if abs(llm_sent) >= 0.2:
                df.at[i, "sentiment_compound"] = llm_sent
            df.at[i, "category"] = res["category"]
            df.at[i, "intent"] = res["intent"]
            df.at[i, "emotional_intensity"] = int(res["emotional_intensity"])
            df.at[i, "short_quote"] = res["short_quote"]
            p, s, w = _infer_flags(r["title"], r["text"], df.at[i,"sentiment_compound"],
                                   GENERIC_QUESTION_TRIGGERS, GENERIC_WORKAROUND_TRIGGERS, GENERIC_PAIN_TRIGGERS)
            df.at[i,"is_pain"], df.at[i,"is_seek"], df.at[i,"is_work"] = p, s, w
        except Exception:
            pass
        if i % 5 == 0:
            time.sleep(0.05 if mode == "fast" else 0.1)
    
    # Compute PVI
    pvi = compute_frs_pvi(df)
    
    # Prepare JSON output
    total = len(df)
    pain_overlap = int(df["is_pain"].sum())
    seek_overlap = int(df["is_seek"].sum())
    work_overlap = int(df["is_work"].sum())
    
    pain_only_n, seek_only_n, work_only_n = exclusive_counts(df)
    
    # Convert DataFrame to JSON-serializable format
    posts_data = []
    for _, row in df.iterrows():
        posts_data.append({
            "title": row["title"],
            "text": row["text"][:500],  # Limit text length
            "subreddit": row["subreddit"],
            "permalink": row["permalink"],
            "category": row["category"],
            "sentiment_compound": float(row["sentiment_compound"]),
            "emotional_intensity": int(row["emotional_intensity"]),
            "short_quote": row["short_quote"],
            "is_pain": bool(row["is_pain"]),
            "is_seek": bool(row["is_seek"]),
            "is_work": bool(row["is_work"]),
            "days_ago": int(row["days_ago"]),
            "created_utc": float(row["created_utc"])
        })
    
    return {
        "assessment": pvi["rating"],
        "pvi": pvi,
        "summary": {
            "total_posts": total,
            "pain_signals": pain_overlap,
            "seek_signals": seek_overlap,
            "work_signals": work_overlap,
            "pain_only": pain_only_n,
            "seek_only": seek_only_n,
            "work_only": work_only_n
        },
        "posts": posts_data[:20]  # Limit to top 20 posts
    }

