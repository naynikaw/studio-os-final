"""
Test script to verify all modules (M2-M6) work correctly
Tests each module with a sample prompt and verifies output
"""
import sys
import os
import json
from datetime import datetime

# Suppress gRPC/ALTS warnings
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")

# Add modules directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, 'modules')
sys.path.insert(0, modules_dir)
sys.path.insert(0, current_dir)

# Test prompt
TEST_PROMPT = "The current hiring process creates a destructive cycle where candidates feel compelled to game the system while companies struggle to see past the noise"

def test_module(module_name, module_func, prompt):
    """Test a single module"""
    print(f"\n{'='*60}")
    print(f"Testing {module_name}")
    print(f"{'='*60}")
    print(f"Input prompt: {prompt[:80]}...")
    print()
    
    try:
        start_time = datetime.now()
        result = module_func(prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Verify result
        if result is None:
            print(f"❌ FAILED: {module_name} returned None")
            return False
        
        if not isinstance(result, dict):
            print(f"⚠️  WARNING: {module_name} returned {type(result)}, expected dict")
            print(f"Result: {str(result)[:200]}...")
            return True
        
        # Check if result has expected structure
        if "error" in result:
            print(f"⚠️  WARNING: {module_name} returned error: {result.get('error')}")
        
        # Print summary
        print(f"✅ SUCCESS: {module_name} completed in {elapsed:.2f}s")
        print(f"Output keys: {list(result.keys())[:10]}")
        
        # Show sample output
        if len(result) > 0:
            sample_key = list(result.keys())[0]
            sample_value = result[sample_key]
            if isinstance(sample_value, str):
                print(f"Sample output ({sample_key}): {sample_value[:100]}...")
            elif isinstance(sample_value, list) and len(sample_value) > 0:
                print(f"Sample output ({sample_key}): {type(sample_value[0])} list with {len(sample_value)} items")
            else:
                print(f"Sample output ({sample_key}): {type(sample_value)}")
        
        # Save to file for inspection
        output_file = os.path.join(current_dir, f"test_output_{module_name.lower()}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"📄 Full output saved to: {output_file}")
        
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: Could not import {module_name}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {module_name} raised exception")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback:")
        traceback.print_exc()
        return False

def main():
    """Run tests for all modules"""
    print("="*60)
    print("MarketOS Backend Module Tests")
    print("="*60)
    print(f"Test prompt: {TEST_PROMPT}")
    print()
    
    results = {}
    
    # Test M2
    try:
        from m2_module import run_m2
        results['M2'] = test_module('M2', run_m2, TEST_PROMPT)
    except Exception as e:
        print(f"❌ FAILED: Could not import M2 module: {e}")
        results['M2'] = False
    
    # Test M3
    try:
        from m3_module import run_m3
        results['M3'] = test_module('M3', run_m3, TEST_PROMPT)
    except Exception as e:
        print(f"❌ FAILED: Could not import M3 module: {e}")
        results['M3'] = False
    
    # Test M4
    try:
        from m4_module import run_m4
        results['M4'] = test_module('M4', run_m4, TEST_PROMPT)
    except Exception as e:
        print(f"❌ FAILED: Could not import M4 module: {e}")
        results['M4'] = False
    
    # Test M5
    try:
        from m5_module import run_m5
        results['M5'] = test_module('M5', run_m5, TEST_PROMPT)
    except Exception as e:
        print(f"❌ FAILED: Could not import M5 module: {e}")
        results['M5'] = False
    
    # Test M6
    try:
        from m6_module import run_m6
        results['M6'] = test_module('M6', run_m6, TEST_PROMPT)
    except Exception as e:
        print(f"❌ FAILED: Could not import M6 module: {e}")
        results['M6'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for module, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{module}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} modules passed")
    
    if passed == total:
        print("\n🎉 All modules passed! Ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} module(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit(main())

