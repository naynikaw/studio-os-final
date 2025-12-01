import os
import requests
import json
import time

# Load key from env or hardcoded fallback (from m2_module.py)
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "c02ff4b7a167f3f52916715952e88f8e9f2359a7e915c44f941cace50df7a5d7")
USER_AGENT = "MarketOS-M2-SerpAPI/1.0"

def fetch_reddit_json(url):
    print(f"Fetching Reddit JSON from: {url}")
    jurl = url if url.endswith(".json") else url + ".json"
    h = {"User-Agent": USER_AGENT}
    
    try:
        r = requests.get(jurl, timeout=25, headers=h)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
                    print(f"Success! Post Title: {post_data.get('title')}")
                    return True
                else:
                    print("Invalid JSON structure")
            except Exception as e:
                print(f"JSON Decode Error: {e}")
        elif r.status_code == 429:
            print("Rate Limited (429)")
        else:
            print(f"Error: {r.text[:200]}")
    except Exception as e:
        print(f"Request Exception: {e}")
    return False

def test_search_and_fetch():
    print(f"Testing SerpAPI with key: {SERPAPI_KEY[:5]}...")
    params = {
        "engine": "google",
        "q": "site:reddit.com hiring assessments cheating",
        "num": 5,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    
    try:
        print("Sending request to SerpAPI...")
        r = requests.get("https://serpapi.com/search", params=params, timeout=15)
        
        if r.status_code == 200:
            data = r.json()
            organic = data.get("organic_results", [])
            print(f"Found {len(organic)} results.")
            
            if organic:
                first_url = organic[0].get('link')
                print(f"Testing fetch on first URL: {first_url}")
                fetch_reddit_json(first_url)
            else:
                print("No organic results found.")
        else:
            print(f"SerpAPI Error: {r.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_search_and_fetch()
