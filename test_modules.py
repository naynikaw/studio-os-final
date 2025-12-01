import sys
import os
import json
import logging

# Add backend to path so we can import modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import modules
try:
    from modules.m2_module import run_m2
    from modules.m3_module import run_m3
    from modules.m4_module import run_m4
    from modules.m5_module import run_m5
    from modules.m6_module import run_m6
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

PROBLEM = "makeup is expensive"
# Use OpenAI API Key from environment
API_KEY = os.getenv("OPENAI_API_KEY")

def test_module(name, run_func, *args, **kwargs):
    print(f"\n{'='*20} Testing {name} {'='*20}")
    try:
        print(f"Running {name} with problem: '{PROBLEM}'...")
        result = run_func(*args, **kwargs)
        
        # Basic validation
        if isinstance(result, dict):
            if "error" in result:
                print(f"❌ {name} returned an error: {result['error']}")
            else:
                print(f"✅ {name} completed successfully.")
                print(f"Output keys: {list(result.keys())}")
                
                # Check for citations
                if "citations" in result:
                    cites = result["citations"]
                    if isinstance(cites, list) and len(cites) > 0:
                        print(f"✅ {name} returned {len(cites)} citations.")
                        print(f"   Sample: {cites[0]}")
                    else:
                        print(f"⚠️ {name} returned empty or invalid citations: {cites}")
                else:
                    print(f"⚠️ {name} missing 'citations' key.")
        else:
            print(f"⚠️ {name} returned unexpected type: {type(result)}")
            
    except Exception as e:
        print(f"❌ {name} failed with exception: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not API_KEY:
        print("⚠️ OPENAI_API_KEY not found in environment. Please set it to run tests.")
        sys.exit(1)

    # Test M2
    test_module("Module 2", run_m2, PROBLEM, api_key=API_KEY)

    # Test M3
    test_module("Module 3", run_m3, PROBLEM, api_key=API_KEY)

    # Test M4
    test_module("Module 4", run_m4, PROBLEM, api_key=API_KEY)

    # Test M5
    test_module("Module 5", run_m5, PROBLEM, api_key=API_KEY)

    # Test M6
    test_module("Module 6", run_m6, PROBLEM, api_key=API_KEY)

if __name__ == "__main__":
    main()
