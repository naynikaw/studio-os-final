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
SERPAPI_KEY      = os.getenv("SERPAPI_KEY")      # https://serpapi.com/
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")   # https://platform.openai.com/
OPENAI_MODEL     = "o4-mini"                     # You can switch models here
# =====================================

# ---- Tunables (kept simple) ----
USER_AGENT = "StudioOS-M2-SerpLite/1.2"
MIN_POST_SCORE = 1
MIN_POST_WORDS = 20
MIN_COMMENT_WORDS = 10
MIN_COMMENT_SCORE = 1
COMMENTS_PER_THREAD = 6
MAX_DAYS_OLD = 365*5
MAX_LLM = 24
RECENCY_HALF_LIFE = 365
F_CAP_DEFAULT = 50
WEIGHTS = (0.40, 0.35, 0.25)

# Heuristic triggers
GENERIC_QUESTION_TRIGGERS = (
    " how ", "how do ", "how can ", " any tips", " tips ", "recommend", "recommendation",
    "suggest", "suggestion", "what should ", "best way", "advice", "any idea", "help?", " help "
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

ASK_PATTERNS = [
    r"\banyone else\b", r"\bwhat do (you|y[’']?all|you guys) use\b",
    r"\blooking for\b", r"\bhow do (you|i)\b", r"^how to\b", r"^what'?s the best\b",
    r"\bbest way\b", r"\brecommendations?\b", r"\bany tips\b"
]
WORK_PATTERNS = [
    r"\bi (used|tried|built|wrote|automated|scripted|fixed)\b",
    r"(^|\n)[\-\*•]\s",
    r"\bhere'?s (what|how) (worked|i did)\b", r"\bstep(?:s)? (i|to)\b"
]

PROMO_BAN = [
    "newsletter","recap","roundup","follow me","sponsor","affiliate",
    "discount","coupon","waitlist","dm me","promo code","hiring","job opening"
]

# ---------- NEW: stopwords + condense helper ----------
STOPWORDS = {
    "the","and","for","with","that","this","from","into","about","their","they","them","only",
    "are","was","were","have","has","had","will","would","can","could","should","a","an","of",
    "to","in","on","at","by","as","it","its","is","be","being","been","or","but","so","if",
    "there","no","not","than","then","when","while","most","more","very","also","such",
    "day","days","time","times","problem","problems","issue","issues","businesses","organizations",
    "companies","teams"
}

def _condense_problem(problem: str) -> Tuple[str, str]:
    text = (problem or "").strip()
    if not text:
        return "", ""

    first_line = ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            first_line = line
            break
    if not first_line:
        first_line = text

    first_line = re.sub(
        r"^problem\s*#?\s*\d*\s*[–\-:]\s*",
        "",
        first_line,
        flags=re.I
    ).strip()

    words = first_line.split()
    if len(words) > 18:
        core_phrase = " ".join(words[:18])
    else:
        core_phrase = first_line

    toks = re.findall(r"[A-Za-z0-9/']+", text.lower())
    key_tokens = []
    seen = set()
    for t in toks:
        if t in STOPWORDS or len(t) <= 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        key_tokens.append(t)
        if len(key_tokens) >= 8:
            break

    keyword_phrase = " ".join(key_tokens).strip()
    if not keyword_phrase:
        keyword_phrase = core_phrase.lower()

    return core_phrase.strip(), keyword_phrase.strip()
# -------------------------------------------------------

# ---------- Utils ----------
URL_RE = re.compile(r"https?://\S+")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
QUOTE_LINE_RE = re.compile(r"^\s*>\s.*$", re.M)

NEGATORS = {"not","no","never","hardly","barely","scarcely","isn’t","can’t","won’t","don’t","doesn’t"}
POS_ADJ = {"good","great","helpful","useful"}
NEG_ADJ = {"bad","awful","terrible","broken","annoying","confusing"}

def utc_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None: return None
    dt = datetime.utcfromtimestamp(ts).replace(tzinfo=tz.UTC).astimezone(tz.tzlocal())
    return dt.isoformat()

def utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def days_ago(ts: float) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days

def clamp(v, a, b): return max(a, min(b, v))
def pct(n, d): return 0 if d == 0 else int(round(100 * n / d))

def short(s, n=220):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n-1] + "…"

def _tok(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _clean_for_sent(text: str) -> str:
    t = URL_RE.sub(" ", text or "")
    t = CODE_FENCE_RE.sub(" ", t)
    t = INLINE_CODE_RE.sub(" ", t)
    t = QUOTE_LINE_RE.sub(" ", t)
    return t

def heuristic_sentiment(title: str, text: str) -> float:
    t = _clean_for_sent(f"{title}\n{text}")
    toks = _tok(t)
    pos = sum(any(p in tok for p in SENT_POS) for tok in toks)
    neg = sum(any(n in tok for n in SENT_NEG) for tok in toks) * 1.2

    for i, tok in enumerate(toks):
        if tok in NEGATORS:
            for j in range(i+1, min(i+4, len(toks))):
                if toks[j] in POS_ADJ:
                    pos = max(0, pos-1); neg += 1
                if toks[j] in NEG_ADJ:
                    neg = max(0, neg-1); pos += 1

    if re.search(r"\b[A-Z]{3,}\b", text or "") or "!!!" in (text or ""):
        if pos > neg: pos += 0.5
        else: neg += 0.5

    denom = pos + neg + 5.0
    return float(clamp((pos - neg) / denom, -1.0, 1.0))

def backoff_sleep(base=0.3, jitter=0.2, attempt=0):
    time.sleep(base * (2 ** attempt) + random.random() * jitter)

def ascii_alpha_ratio(s: str) -> float:
    if not s: return 1.0
    letters = sum(ch.isalpha() and ord(ch) < 128 for ch in s)
    total = max(1, sum(ch != " " for ch in s))
    return letters / total

def _ngrams(text: str, n=3):
    toks = _tok(text)
    return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set()

def jaccard_3gram(a: str, b: str) -> float:
    A, B = _ngrams(a, 3), _ngrams(b, 3)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# ---------- SerpApi search + Reddit fetch ----------
def serpapi_search(query: str, max_results: int) -> List[str]:
    if not SERPAPI_KEY or "PASTE_YOUR_SERPAPI_KEY" in SERPAPI_KEY:
        raise SystemExit("Fill SERPAPI_KEY at top.")
    params = {
        "engine": "google",
        "q": query,
        "num": 10,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=30,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()
    urls: List[str] = []
    for item in data.get("organic_results", []):
        u = item.get("link")
        if u and re.search(r"(reddit\.com|redd\.it)", u):
            urls.append(u)
            if len(urls) >= max_results:
                break
    return urls

def normalize_reddit_url(url: str) -> str:
    if re.match(r"https?://(www\.)?redd\.it/[A-Za-z0-9]+/?", url):
        return url.rstrip("/")
    m = re.match(r"https?://(www\.|old\.|np\.)?reddit\.com(/r/[^/]+)?/comments/[a-z0-9]{6,8}/[^/?#]*/?", url)
    if m: return m.group(0).rstrip("/")
    m2 = re.match(r"https?://(www\.|old\.|np\.)?reddit\.com/comments/[a-z0-9]{6,8}/?", url)
    if m2: return m2.group(0).rstrip("/")
    return url.rstrip("/")

def fetch_reddit_json(url: str, max_attempts=3) -> Optional[List[Any]]:
    jurl = url if url.endswith(".json") else url + ".json"
    h = {"User-Agent": USER_AGENT}
    for attempt in range(max_attempts):
        try:
            r = requests.get(jurl, timeout=25, headers=h)
        except Exception as e:
            print(f"[M2] Reddit fetch error for {jurl}: {e}")
            return None

        if r.status_code == 429:
            print(f"[M2] Reddit rate limited (429) for {jurl}, attempt {attempt+1}")
            backoff_sleep(attempt=attempt)
            continue

        if not r.ok:
            # THIS is what we want to see in Railway logs
            print(f"[M2] Reddit fetch failed: {r.status_code} {r.reason} for {jurl}")
            return None

        try:
            return r.json()
        except Exception as e:
            print(f"[M2] Reddit JSON parse error for {jurl}: {e}")
            backoff_sleep(attempt=attempt)

    return None


def extract_post(payload: List[Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    try:
        sd = payload[0]["data"]["children"][0]["data"]
    except Exception:
        return None

    title = (sd.get("title") or "").strip()
    text = (sd.get("selftext") or "").strip()
    hay = f"{title}\n{text}".lower()

    if any(p in hay for p in PROMO_BAN):
        return None

    if len((title + " " + text).split()) < MIN_POST_WORDS:
        return None
    if int(sd.get("score", 0)) < MIN_POST_SCORE:
        return None
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

# ---------- LLM adapter ----------
class _OpenAIAdapter:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    class _Resp:
        def __init__(self, text: str):
            self.text = text

    def generate_content(self, parts: List[str]):
        prompt = "\n\n".join(str(p) for p in parts if p is not None)
        try:
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt
            )
            text = getattr(resp, "output_text", None)
            if not text:
                try:
                    blocks = []
                    for item in getattr(resp, "output", []) or []:
                        if getattr(item, "type", "") == "message":
                            for c in getattr(item, "content", []) or []:
                                if getattr(c, "type", "") == "output_text":
                                    blocks.append(getattr(c, "text", ""))
                    text = "\n".join(blocks).strip()
                except Exception:
                    text = ""
            return _OpenAIAdapter._Resp(text or "")
        except Exception:
            return _OpenAIAdapter._Resp("")

def init_gemini():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        raise SystemExit("Fill OPENAI_API_KEY at top or set env OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI()
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

def run_m2(prompt: str, max_results: int = 10, include_report: bool = False) -> Dict[str, Any]:
    df, pvi = _analyze(prompt, max_results)

    if df.empty:
        out = {
            "summary": {
                "problem": prompt,
                "total": 0,
                "pain": 0, "seek": 0, "work": 0,
                "pain_pct": 0, "opportunity_pct": 0
            },
            "pvi": pvi,
            "window": None,
            "highlights": {"pain": [], "seek": [], "work": []}
        }
        if include_report:
            out["report_text"] = "No qualifying Reddit content for this query and thresholds."
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
