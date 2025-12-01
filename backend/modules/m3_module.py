"""
Module M3: Problem Understanding & Cost Analysis
Agent-based with autonomous tool use and conversation support
"""
import os
import json
import logging
import sys
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

import arxiv

# Configure OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
try:
    from openai import OpenAI
    # Use standard OpenAI endpoint
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")
except ImportError:
    logger.error("openai package not installed - please install it")
    raise
except Exception as e:
    logger.error(f"Failed to initialize OpenAI: {e}")
    raise

# Initialize Tavily client (only for cost analysis)
tavily_api_key = os.getenv("TAVILY_API_KEY", "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")
tavily_client = None
try:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=tavily_api_key)
    logger.info("Tavily client initialized (for cost analysis only)")
except ImportError:
    logger.info("tavily-python not installed - Tavily will not be available")
except Exception as e:
    error_str = str(e).lower()
    if "api key" in error_str or "unauthorized" in error_str or "invalid" in error_str:
        logger.warning("Tavily API key invalid - Tavily will not be available")
        tavily_client = None
    else:
        logger.warning(f"Failed to initialize Tavily: {e}")
        tavily_client = None

# ==================== Tool Functions ====================

def search_arxiv(query: str, max_results: int = 2, category: Optional[str] = None, sort: str = "relevance") -> dict:
    """Search arXiv for academic papers. Returns top 2 most relevant results based on relevance ranking."""
    try:
        logger.info(f"Searching arXiv: query='{query}', max_results={max_results}, sort={sort}")
        
        search_query = query
        if category:
            search_query = f"cat:{category} AND {query}"
        
        # Use relevance as default for quality - arXiv's algorithm ranks by relevance
        sort_criterion = arxiv.SortCriterion.Relevance if sort == "relevance" else arxiv.SortCriterion.SubmittedDate
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        results = []
        for paper in search.results():
            results.append({
                "arxiv_id": paper.entry_id.split('/')[-1],
                "title": paper.title,
                "authors": [author.name for author in paper.authors[:3]],
                "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                "published": paper.published.isoformat(),
                "pdf_url": paper.pdf_url
            })
        
        logger.info(f"Found {len(results)} arXiv papers")
        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}", exc_info=True)
        return {"success": False, "error": str(e), "results": []}

def get_arxiv_paper(arxiv_id: str) -> dict:
    """Fetch full metadata and PDF URL for a specific arXiv paper."""
    try:
        logger.info(f"Fetching arXiv paper: {arxiv_id}")
        paper = next(arxiv.Search(id_list=[arxiv_id]).results(), None)
        
        if not paper:
            return {"success": False, "error": f"Paper {arxiv_id} not found", "data": None}
        
        return {
            "success": True,
            "data": {
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "authors": [{"name": author.name} for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.isoformat(),
                "pdf_url": paper.pdf_url,
                "categories": paper.categories
            }
        }
    except Exception as e:
        logger.error(f"Error fetching arXiv paper: {e}", exc_info=True)
        return {"success": False, "error": str(e), "data": None}

def tavily_search(query: str, max_results: int = 5) -> dict:
    """General-purpose web search using Tavily. Use for cost analysis queries and affected segments research. Returns top 5 results sorted by quality score (relevance)."""
    if not tavily_client:
        return {"success": False, "error": "Tavily not available", "results": []}
    
    try:
        # Request more results to ensure we get top quality ones after sorting
        request_max = max(max_results * 2, 10)
        logger.info(f"Tavily search (cost analysis): query='{query}', max_results={max_results} (requesting {request_max} for quality filtering)")
        response = tavily_client.search(query=query, max_results=request_max, search_depth="advanced")
        
        # Get all results with scores
        all_results = []
        for result in response.get("results", []):
            content = result.get("content", "")
            score = result.get("score", 0.0)
            all_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content[:500] + "..." if len(content) > 500 else content,
                "score": score,
                "quality_score": score  # Explicit quality metric
            })
        
        # Sort by quality score (relevance) descending, then take top N
        all_results.sort(key=lambda x: x["score"], reverse=True)
        results = all_results[:max_results]
        
        logger.info(f"Found {len(results)} top-quality Tavily results (sorted by relevance score)")
        if results:
            avg_quality = sum(r["score"] for r in results) / len(results)
            logger.info(f"Average quality score: {avg_quality:.3f}")
        
        return {
            "success": True, 
            "count": len(results), 
            "results": results,
            "quality_metric": "relevance_score",  # Quality metric used
            "avg_quality_score": sum(r["score"] for r in results) / len(results) if results else 0.0
        }
    except Exception as e:
        # Handle API key errors gracefully
        error_str = str(e).lower()
        if "unauthorized" in error_str or "api key" in error_str or "invalid" in error_str:
            logger.warning(f"Tavily API key issue - skipping web search: {type(e).__name__}")
            return {"success": False, "error": "Tavily API key invalid or missing", "results": []}
        else:
            logger.warning(f"Tavily search failed: {type(e).__name__}: {str(e)[:100]}")
            return {"success": False, "error": str(e)[:100], "results": []}

# ==================== Tool Definitions for OpenAI Function Calling ====================

def execute_tool(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool function based on function name and arguments"""
    logger.info(f"Executing tool: {function_name} with args: {arguments}")
    
    if function_name == "search_arxiv":
        return search_arxiv(
            query=arguments.get("query", ""),
            max_results=arguments.get("max_results", 2),
            category=arguments.get("category"),
            sort=arguments.get("sort", "relevance")
        )
    elif function_name == "get_arxiv_paper":
        return get_arxiv_paper(arxiv_id=arguments.get("arxiv_id", ""))
    elif function_name == "tavily_search":
        return tavily_search(
            query=arguments.get("query", ""),
            max_results=arguments.get("max_results", 5)  # Default to top 5 for enriched data
        )
    else:
        return {"success": False, "error": f"Unknown function: {function_name}"}

# ==================== OpenAI Agent Functions ====================

def get_openai_tools() -> List[Dict[str, Any]]:
    """Convert our tool definitions to OpenAI function calling format"""
    tools = []
    
    # search_arxiv
    tools.append({
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Use this for academic papers, especially arXiv references. Search arXiv database for research papers on a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for academic papers"},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default: 2 for top quality results)", "default": 2},
                    "category": {"type": "string", "description": "Optional arXiv category (e.g., 'cs.AI', 'econ.EM')"},
                    "sort": {"type": "string", "enum": ["recent", "relevance"], "description": "Sort order: 'relevance' for most relevant (recommended), 'recent' for newest", "default": "relevance"}
                },
                "required": ["query"]
            }
        }
    })
    
    # get_arxiv_paper
    tools.append({
        "type": "function",
        "function": {
            "name": "get_arxiv_paper",
            "description": "Fetch full metadata and PDF URL for a specific arXiv paper by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g., '2301.12345')"}
                },
                "required": ["arxiv_id"]
            }
        }
    })
    
    # tavily_search (if available)
    if tavily_client:
        tools.append({
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "Use this for: (1) Quantitative cost data - search 'cost of [problem]', 'ROI of solving [problem]', salary data, time studies, economic statistics (BLS, Census), analyst reports (Gartner, Forrester). (2) Qualitative costs - 'employee morale and [problem]', 'customer frustration with [problem]', productivity impacts. (3) Affected populations - research GENERAL PEOPLE/ROLES/DEPARTMENTS only (e.g., 'AI/ML engineers', 'Data scientists', 'Product managers', 'HR professionals', 'IT staff', 'HR Services', 'IT Department', 'Operations Teams', 'Customer Service', 'Sales Teams', 'Finance Department', 'Product Development'). Do NOT use industry-specific combinations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for cost analysis data"},
                        "max_results": {"type": "integer", "description": "Maximum number of results (default: 5 for enriched data, sorted by relevance score)", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        })
    
    return tools

def run_agent_pipeline(
    problem: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_context: Optional[str] = None
) -> str:
    """OpenAI agent pipeline with function calling"""
    if not openai_client:
        raise Exception("OpenAI client not available")
    
    # Parse user context if provided
    persona_info = ""
    if user_context:
        try:
            context_data = json.loads(user_context)
            persona = context_data.get("persona", "")
            primary_goal = context_data.get("primary_goal", "")
            secondary_goal = context_data.get("secondary_goal", "")
            focus_area = context_data.get("focus_area", "")
            
            if persona:
                persona_info = f"\n\nUser Persona Context:\n"
                persona_info += f"- Persona: {persona}\n"
                if primary_goal:
                    persona_info += f"- Primary Goal: {primary_goal}\n"
                if secondary_goal:
                    persona_info += f"- Secondary Goal: {secondary_goal}\n"
                if focus_area:
                    persona_info += f"- Focus Area: {focus_area}\n"
                persona_info += "\nPlease tailor your analysis to align with this user's persona and goals.\n"
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse user_context: {e}")
    
    # Build system message - Management Consultant Approach
    system_message = """You are a management consultant creating a "Deep Dive Report" that explains root causes and quantifies costs to build a business case for investment.

PRIMARY OBJECTIVE: Create a logical, data-supported argument that demonstrates this problem is costing a specific audience meaningful time, money, or strategic advantage.

STRATEGIC MINDSET: Think like a management consultant building a business case. Move beyond surface-level pain and construct a logical argument supported by data.

**CRITICAL INSTRUCTION: CITATIONS**
- You MUST provide in-text citations for every claim you make, e.g., "The market is growing at 5% [1]" or "Smith et al. found that... [2]".
- At the end of your analysis, you MUST provide a "References" section listing all sources used.
- When using tools, keep track of the URLs/Titles and ensure they are reflected in the final output.

STEP-BY-STEP TACTICAL PROCESS:

1. MULTI-LAYERED RESEARCH:

Root Cause Analysis: Use a "5 Whys" mental model internally. Start with the problem and ask "Why?" five times to drill down to root causes. Use search_arxiv to find academic studies, industry white papers, and investigative journalism that explore the history and contributing factors. Your goal is to identify the fundamental root causes. In your final output, present each root cause as a CLEAR, CONCISE STATEMENT (not with arrows or "Why?" chains). Each root cause should be a well-written sentence that explains WHY it leads to the problem. Use external knowledge from search APIs - do NOT rely on generic knowledge alone.

Quantitative Cost Search: Use tavily_search to find benchmarks and data points. Search for "cost of [problem]", "ROI of solving [problem]", "time spent on [related task]". Look for reports from respected analysts (Gartner, Forrester), government agencies (BLS, Census), and university studies. Use external knowledge from search APIs - do NOT rely on generic estimates.

Qualitative Cost Search: Use tavily_search to identify costs not easily measured in dollars. Search for "employee morale and [problem]", "customer frustration with [problem]", "productivity impact of [problem]". Use external knowledge from search APIs to find real evidence.

Affected Populations Research: Use tavily_search to research which PEOPLE/ROLES/DEPARTMENTS are most impacted. Use external knowledge from search APIs - do NOT rely on generic knowledge. Think about who is DIRECTLY affected by this specific problem. Focus on general people/roles/departments such as: "AI/ML engineers", "Data scientists", "Product managers", "HR professionals", "IT staff", "Operations teams", "Customer service reps", "Sales teams", "Finance professionals", "Product developers", "HR Services", "IT Department", "Operations Teams", "Customer Service", "Sales Teams", "Finance Department", "Product Development". Do NOT use industry-specific combinations. Research using external APIs to find real data.

2. COST MODELING & SYNTHESIS:

Construct a transparent "Cost of Inaction" model. State your formula clearly. Example: (Avg. Hours Wasted per Employee per Week) x (Number of Employees) x (Avg. Hourly Wage) x (52 Weeks) = Annual Cost.

Always state your assumptions clearly. If you can't find a direct data point, make a reasonable, conservative estimate and label it as such (e.g., "Assuming a conservative estimate of 2 hours wasted per week...").

Pair quantitative data with qualitative evidence. Example: "The estimated $1.2M annual cost in lost productivity is compounded by a documented decrease in employee morale, evidenced by discussions on Glassdoor."

3. INTERNAL VALIDATION:

Assumption Check: Are assumptions clearly stated and defensible?
Source Quality Check: Are sources credible and cited? Use primary or respected secondary sources.
Narrative Check: Does the report tell a clear story from root causes to tangible impact?

TOOL USAGE RULES:
- ALWAYS use search_arxiv for root cause analysis, historical context, and academic research
- Use tavily_search for: (1) Quantitative cost data: "cost of [problem]", salary data, time studies, ROI reports; (2) Qualitative costs: employee morale, customer satisfaction, productivity impacts; (3) Affected populations: research general people/roles/departments (e.g., "AI/ML engineers", "Data scientists", "HR Services", "IT Department", "Operations Teams", "Customer Service", "Sales Teams", "Finance Department", "Product Development")
- DO NOT use tavily_search for theoretical frameworks - use search_arxiv instead
- Use get_arxiv_paper when you need full details of a specific paper

CRITICAL: You MUST return your analysis in EXACTLY this JSON format - a "Deep Dive Report" structure:
{
  "executive_summary": "A one-paragraph summary of the problem and its total estimated cost. This should be compelling and immediately show the 'so what'.",
  "root_causes": [
    "Root cause 1 - A clear, concise statement of the root cause. Use '5 Whys' analysis internally to identify it, but present it as a clean statement showing WHY it leads to the problem. Supported by research from search APIs.",
    "Root cause 2 - A clear, concise statement of the root cause. Supported by research from search APIs.",
    "Root cause 3 - A clear, concise statement of the root cause. Supported by research from search APIs."
  ],
  "cost_analysis": {
    "qualitative_costs": [
      "Qualitative cost 1 (e.g., employee morale, customer frustration) with specific evidence",
      "Qualitative cost 2 with specific evidence",
      "Qualitative cost 3 with specific evidence"
    ],
    "quantitative_estimates": {
      "annual_cost_range": "Estimated range (e.g., $10M-$50M annually) - be specific",
      "methodology": "Your cost formula clearly stated. Example: (Hours Wasted per Week) x (Employees) x (Hourly Wage) x (52) = Annual Cost. Cite sources.",
      "assumptions": [
        "Assumption 1 - clearly stated and defensible",
        "Assumption 2 - clearly stated and defensible",
        "Assumption 3 - if applicable"
      ]
    }
  },
  "affected_segments": [
    {
      "segment": "People/role/department - general people/roles/departments (e.g., 'AI/ML engineers', 'Data scientists', 'Product managers', 'HR professionals', 'IT staff', 'HR Services', 'IT Department', 'Operations Teams', 'Customer Service', 'Sales Teams', 'Finance Department', 'Product Development')",
      "impact": "Specific description of how these people/roles/departments are impacted. Include quantitative data when available, specific examples, and concrete ways the problem manifests. 2-4 sentences with rich details from research."
    },
    {
      "segment": "Another people/role/department (general only, no industry-specific)",
      "impact": "Rich, detailed description with specific details from research"
    },
    {
      "segment": "Yet another people/role/department (general only, no industry-specific)",
      "impact": "Rich, detailed description with specific details from research"
    }
  ],
  "citations": [
    "Source 1 - credible source",
    "Source 2 - credible source"
  ],
  "confidence_score": 0.85
}

CRITICAL REQUIREMENTS:
- Root Causes: MUST use "5 Whys" analysis internally to identify root causes, but present them as CLEAR, CONCISE STATEMENTS in the output (no arrows, no "Why?" chains). Each root cause should be a well-written sentence that explains WHY it leads to the problem. Use search_arxiv to find external knowledge - do NOT rely on generic knowledge.
- Affected Segments: Think about who is DIRECTLY affected by this specific problem. Use tavily_search to research using external knowledge. Use ONLY general people/roles/departments such as: "AI/ML engineers", "Data scientists", "Product managers", "HR professionals", "IT staff", "HR Services", "IT Department", "Operations Teams", "Customer Service", "Sales Teams", "Finance Department", "Product Development". Do NOT use industry-specific combinations (e.g., NOT "IT Department in healthcare", just "IT Department" or "IT staff").
- External Knowledge: ALWAYS use search_arxiv and tavily_search to gather external knowledge. Do NOT rely on generic LLM knowledge alone. Cite your sources.
- Cost Analysis: Use external data from search APIs. State your formula clearly and cite sources for your assumptions.

Return ONLY valid JSON, no additional text."""
    
    # Check if this is a follow-up question
    is_followup = conversation_history is not None and len(conversation_history) > 0
    
    # Use different system prompts for initial vs follow-up
    if is_followup:
        # Conversational system prompt for follow-ups
        followup_system_message = """You are a helpful assistant having a conversation about a business problem analysis. 
        
The user has already received a Deep Dive Report about a problem. They are now asking follow-up questions.

Provide clear, conversational answers based on the previous analysis. Be helpful, concise, and directly address their question. 
Do NOT return JSON or structured reports - just provide a natural conversational response.

If they ask about specific parts of the analysis (like cost methodology, root causes, affected segments), reference the previous analysis and provide additional context or clarification."""
        messages = [{"role": "system", "content": followup_system_message + persona_info}]
    else:
        # Full system prompt for initial requests
        messages = [{"role": "system", "content": system_message + persona_info}]
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                # Convert assistant responses to conversational string format
                if isinstance(content, dict):
                    # If it's a dict (from previous response), convert to conversational summary
                    summary = content.get("executive_summary", "")
                    if summary:
                        messages.append({"role": "assistant", "content": summary})
                    else:
                        messages.append({"role": "assistant", "content": str(content)})
                elif isinstance(content, str):
                    # Use as-is if it's already a string
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "assistant", "content": str(content)})
            else:
                messages.append({"role": "user", "content": str(content)})
    
    # Add current problem
    if not is_followup:
        # Initial request - full Deep Dive Report
        messages.append({"role": "user", "content": f"Problem: {problem}\n\nCreate a Deep Dive Report using EXTERNAL KNOWLEDGE from search APIs. Use search_arxiv for root cause analysis - apply '5 Whys' internally to identify root causes, but present them as CLEAR, CONCISE STATEMENTS (no arrows or 'Why?' chains in the output). Use tavily_search for: (1) Quantitative cost data - search 'cost of [problem]', 'ROI of solving [problem]', time studies; (2) Qualitative costs - employee morale, customer frustration, productivity impacts; (3) Affected populations - research who is DIRECTLY affected. Use ONLY general people/roles/departments such as: 'AI/ML engineers', 'Data scientists', 'Product managers', 'HR professionals', 'IT staff', 'HR Services', 'IT Department', 'Operations Teams', 'Customer Service', 'Sales Teams', 'Finance Department', 'Product Development'. Do NOT use industry-specific combinations. Make multiple search calls to gather external knowledge. Do NOT rely on generic knowledge - use search APIs to find real data and research."})
    else:
        # Follow-up question - conversational response, use tools only if needed
        messages.append({"role": "user", "content": problem})
    
    max_iterations = 1  # Reduced to 1 since we're not using tools - single OpenAI call
    # TOOLING COMMENTED OUT - Using direct OpenAI calls only
    iteration = 0
    final_response = None  # Initialize to avoid UnboundLocalError
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Agent iteration {iteration}")
        
        try:
            # TOOLING COMMENTED OUT - Using direct OpenAI calls only
            # For follow-ups, only provide tools if the question might need external data
            # Otherwise, just use OpenAI directly for conversational response
            # if is_followup:
            #     # Check if follow-up question might need external data
            #     needs_tools = any(keyword in problem.lower() for keyword in [
            #         "cost", "data", "research", "study", "report", "statistics", 
            #         "impact", "affected", "segments", "citations", "source"
            #     ])
            #     if needs_tools:
            #         response = openai_client.chat.completions.create(
            #             model="gemini-2.5-pro",
            #             messages=messages,
            #             tools=get_openai_tools(),
            #             tool_choice="auto",
            #             temperature=0.7
            #         )
            #     else:
            #         # Simple conversational response without tools
            #         response = openai_client.chat.completions.create(
            #             model="gemini-2.5-pro",
            #             messages=messages,
            #             temperature=0.7
            #         )
            # else:
            #     # Initial request - always use tools
            #     response = openai_client.chat.completions.create(
            #         model="gemini-2.5-pro",
            #         messages=messages,
            #         tools=get_openai_tools(),
            #         tool_choice="auto",
            #         temperature=0.7
            #     )
            
            # Direct OpenAI call without tools
            response = openai_client.chat.completions.create(
                model="gpt-5.1",
                messages=messages,
                temperature=0.7
            )
            
            message = response.choices[0].message
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": message.content
                # TOOLING COMMENTED OUT
                # "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])]
            })
            
            # TOOLING COMMENTED OUT - Direct response only
            # # Check for function calls
            # if message.tool_calls:
            #     # Execute all tool calls first
            #     for tool_call in message.tool_calls:
            #         function_name = tool_call.function.name
            #         arguments = json.loads(tool_call.function.arguments)
            #         
            #         logger.info(f"OpenAI requested tool: {function_name}")
            #         tool_result = execute_tool(function_name, arguments)
            #         
            #         messages.append({
            #             "role": "tool",
            #             "tool_call_id": tool_call.id,
            #             "content": json.dumps(tool_result, indent=2)
            #         })
            #     
            #     # If this is the last iteration, force a final response without more tool calls
            #     if iteration >= max_iterations:
            #         logger.info(f"Last iteration reached - forcing final response after tool execution")
            #         # Make one final call without tools to get a response
            #         final_response_obj = openai_client.chat.completions.create(
            #             model="gpt-4o",
            #             messages=messages,
            #             temperature=0.7
            #         )
            #         final_response = final_response_obj.choices[0].message.content.strip()
            #         break
            #     
            #     # Not last iteration - continue to next iteration
            #     continue
            
            # Get final response directly (no tool calls)
            final_response = message.content.strip() if message.content else "No response generated"
            break
            
        except Exception as e:
            logger.error(f"Error in OpenAI agent iteration: {e}", exc_info=True)
            final_response = f"Error during analysis: {str(e)}"
            break
    
    if not final_response:
        final_response = "Unable to generate response after maximum iterations."
    
    # For follow-ups, return simple conversational response; for initial requests, return structured JSON
    if is_followup:
        # Follow-up: return simple JSON with just the response string
        return json.dumps({
            "response": final_response
        })
    else:
        # Initial request: Extract JSON from structured response
        import re
        json_match = re.search(r'\{.*\}', final_response, flags=re.DOTALL)
        if json_match:
            return json_match.group(0)
        else:
            return json.dumps({
                "executive_summary": final_response[:200] if final_response else f"Analysis of: {problem}",
                "root_causes": [],
                "cost_analysis": {
                    "qualitative_costs": [],
                    "quantitative_estimates": {
                        "annual_cost_range": "To be determined",
                        "methodology": "LLM-based analysis with tool enrichment",
                        "assumptions": []
                    }
                },
                "affected_segments": [],
                "citations": [],
                "confidence_score": 0.7
            })


def run_pipeline(problem: str, user_context: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Core pipeline function - now uses agent-based approach.
    """
    return run_agent_pipeline(problem, conversation_history=conversation_history, user_context=user_context)

def run_m3(problem: str, user_context: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None, api_key: str = None) -> dict:
    """
    Main function for M3 module with conversational agent support.
    Returns original format for initial requests, wrapped in "response" for follow-ups.
    """
    # Use passed API key if available
    global openai_client
    if api_key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to re-initialize OpenAI with passed key: {e}")

    try:
        # Check if this is a follow-up
        is_followup = conversation_history is not None and len(conversation_history) > 0
        
        # Run the agent pipeline
        json_str = run_pipeline(problem, user_context=user_context, conversation_history=conversation_history)
        
        # Parse JSON string to dict
        if isinstance(json_str, str):
            # Robust extraction: find first { and last }
            try:
                start = json_str.find('{')
                end = json_str.rfind('}')
                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
                response_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: try to parse as-is or return error
                logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
                response_data = {"response": json_str} # Fallback for chat
        else:
            response_data = json_str
        
        # For follow-ups, response_data is already {"response": "..."}, return as-is
        # For initial requests, return original format
        if is_followup:
            # Follow-up response is already in {"response": "..."} format
            return response_data

        else:
            # Initial request: return original format
            return response_data
    except Exception as e:
        logger.error(f"Error in run_m3: {e}", exc_info=True)
        # Return error in original format
        error_response = {
            "executive_summary": f"Error processing problem: {str(e)}",
            "root_causes": [],
            "cost_analysis": {
                "qualitative_costs": [],
                "quantitative_estimates": {
                    "annual_cost_range": "",
                    "methodology": "",
                    "assumptions": []
                }
            },
            "affected_segments": [],
            "citations": [],
            "confidence_score": 0.0
        }
        # For follow-ups, return simple error response; for initial requests, return full structure
        if conversation_history is not None and len(conversation_history) > 0:
            return {"response": f"Error processing problem: {str(e)}"}
        else:
            return error_response

