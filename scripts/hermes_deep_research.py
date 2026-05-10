#!/usr/bin/env python3
"""
Hermes integration wrapper for Open Deep Research agent.
Usage:
  python scripts/hermes_deep_research.py "What is the latest in AI agents?"
  python scripts/hermes_deep_research.py --model "deepseek:deepseek-chat" "Research quantum computing advances"
"""
import argparse, json, os, sys, time
from pathlib import Path

# Add project to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# Load API keys from Hermes env
HERMES_ENV = Path("/opt/data/.env")
if HERMES_ENV.exists():
    with open(HERMES_ENV) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                if key in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                    os.environ.setdefault(key, val)

def run_research(query: str, model: str = "deepseek:deepseek-chat", search_api: str = "none",
                 max_iterations: int = 4, max_concurrent: int = 3) -> dict:
    """Run the deep research agent and return results."""
    
    from open_deep_research.deep_researcher import deep_researcher
    from open_deep_research.configuration import Configuration, SearchAPI
    
    config = Configuration(
        research_model=model,
        summarization_model=model,
        compression_model=model,
        final_report_model=model,
        search_api=SearchAPI(search_api),
        max_researcher_iterations=max_iterations,
        max_concurrent_research_units=max_concurrent,
        allow_clarification=False,
    )
    
    start = time.time()
    
    # Compile with config
    app = deep_researcher
    
    # Run the graph
    initial_state = {
        "messages": [{"role": "user", "content": query}],
    }
    
    result = app.invoke(
        initial_state,
        config={"configurable": config.model_dump()}
    )
    
    elapsed = time.time() - start
    
    # Extract final report
    final_report = ""
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            final_report += str(msg.content) + "\n"
    
    return {
        "ok": True,
        "query": query,
        "model": model,
        "elapsed_seconds": round(elapsed, 1),
        "final_report": final_report.strip(),
        "message_count": len(result.get("messages", [])),
    }

def main():
    parser = argparse.ArgumentParser(description="Hermes Deep Research Agent")
    parser.add_argument("query", help="Research question")
    parser.add_argument("--model", default="deepseek:deepseek-chat", help="LLM model to use")
    parser.add_argument("--search", default="none", choices=["tavily", "openai", "anthropic", "none"])
    parser.add_argument("--iterations", type=int, default=4, help="Max research iterations")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent research units")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    result = run_research(
        query=args.query,
        model=args.model,
        search_api=args.search,
        max_iterations=args.iterations,
        max_concurrent=args.concurrent,
    )
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"🔬 Deep Research: {args.query}")
        print(f"   Model: {args.model} | Search: {args.search}")
        print(f"   Time: {result['elapsed_seconds']}s | Messages: {result['message_count']}")
        print(f"{'='*60}\n")
        print(result["final_report"])
        print(f"\n{'='*60}")
        print(f"✅ Research complete in {result['elapsed_seconds']}s")

if __name__ == "__main__":
    main()
