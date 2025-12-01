"""
Module M6: Market Analysis & Competitive Intelligence
Refactored for FastAPI integration
"""
import sys
import os

# Suppress gRPC/ALTS warnings
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")

# Add parent directory to path to import from m6.py
# Check if m6.py is in parent directory (studio-chain-flow-main)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
studio_dir = os.path.join(grandparent_dir, 'studio-chain-flow-main')
sys.path.insert(0, studio_dir)
sys.path.insert(0, parent_dir)

# Import the analyze_market function from m6.py
try:
    from m6 import analyze_market, render_report_md
    import json
except ImportError:
    # Fallback: try importing from file directly
    import importlib.util
    m6_path = os.path.join(studio_dir, 'm6.py')
    if os.path.exists(m6_path):
        spec = importlib.util.spec_from_file_location("m6", m6_path)
        m6_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m6_module)
        analyze_market = m6_module.analyze_market
        render_report_md = getattr(m6_module, 'render_report_md', None)
    else:
        analyze_market = None
        render_report_md = None

def run_m6(problem: str) -> dict:
    """
    Main function for M6 module.
    Returns JSON-serializable dict with market analysis.
    """
    try:
        if analyze_market is None:
            raise ImportError("Could not import analyze_market from m6.py")
        
        # Extract concept name and other details from problem
        # For now, use defaults
        concept_name = os.getenv("CONCEPT_NAME", "GoodFutures")
        icp = os.getenv("ICP", "VC analysts, corp strategy, consulting")
        market_topic = os.getenv("MARKET_TOPIC", "market & competitive intelligence software")
        
        # Run the analysis
        report = analyze_market(
            concept_name=concept_name,
            problem=problem,
            icp=icp,
            alt_solutions=["PitchBook", "CB Insights", "AlphaSense", "Crunchbase"],
            market_topic=market_topic
        )
        
        # Convert report to dict if needed
        if isinstance(report, dict):
            return report
        else:
            # Try to parse as JSON string
            try:
                return json.loads(str(report))
            except:
                return {
                    "report_type": "Market Analysis",
                    "problem": problem,
                    "raw_report": str(report)
                }
    except Exception as e:
        return {
            "error": str(e),
            "report_type": "Market Analysis",
            "problem": problem,
            "sections": {}
        }

