#!/usr/bin/env python3
"""
Open Deep Research wired to Modal Ollama (Nemotron 3 33B).

Usage:
  # One-shot research
  python scripts/research_with_nemotron.py "What are the latest AI agent frameworks?"
  
  # Autonomous agent mode — watches a queue file and processes tasks
  python scripts/research_with_nemotron.py --watch
  
  # Batch process from JSONL
  python scripts/research_with_nemotron.py --batch tasks.jsonl
"""
import argparse, json, os, sys, time, asyncio
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# Default Ollama endpoint on Modal
OLLAMA_BASE = os.environ.get(
    "NEMOTRON_OLLAMA_URL",
    "https://m1ndb0t-2045--mindexpander-nemotron-ollama-nemotronservi-be5f8e.modal.run"
)
MODEL_NAME = "nemotron3:33b"

# Output directory for research reports
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def ollama_chat(messages: list, max_tokens: int = 4096, temperature: float = 0.7) -> str:
    """Call Ollama chat API (OpenAI-compatible)."""
    import requests
    resp = requests.post(
        f"{OLLAMA_BASE}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_research(query: str, max_iterations: int = 3, search_api: str = "none") -> dict:
    """Run Open Deep Research with Nemotron as the backend."""
    
    # Set up LangChain to use our Ollama endpoint as an OpenAI-compatible API
    os.environ["OPENAI_API_KEY"] = "ollama"  # Ollama doesn't need a real key
    os.environ["OPENAI_BASE_URL"] = f"{OLLAMA_BASE}/v1"
    
    from open_deep_research.deep_researcher import deep_researcher
    from open_deep_research.configuration import Configuration, SearchAPI
    
    config = Configuration(
        research_model=f"openai:{MODEL_NAME}",
        summarization_model=f"openai:{MODEL_NAME}",
        compression_model=f"openai:{MODEL_NAME}",
        final_report_model=f"openai:{MODEL_NAME}",
        search_api=SearchAPI(search_api),
        max_researcher_iterations=max_iterations,
        max_concurrent_research_units=2,
        allow_clarification=False,
        max_react_tool_calls=5,
    )
    
    start = time.time()
    initial_state = {"messages": [{"role": "user", "content": query}]}
    
    async def _run():
        return await deep_researcher.ainvoke(
            initial_state,
            config={"configurable": config.model_dump()}
        )
    
    result = asyncio.run(_run())
    elapsed = time.time() - start
    
    # Extract final report
    final_report = ""
    messages_list = result.get("messages", [])
    for msg in messages_list:
        if hasattr(msg, "content") and msg.content:
            content = str(msg.content)
            if len(content) > 200:  # skip short tool messages
                final_report = content  # last long message is the report
    
    return {
        "ok": True,
        "query": query,
        "model": MODEL_NAME,
        "elapsed_seconds": round(elapsed, 1),
        "final_report": final_report,
        "message_count": len(messages_list),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_report(result: dict):
    """Save research report to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = result["query"][:50].lower().replace(" ", "_").replace("?", "").replace("/", "_")
    filename = f"research_{timestamp}_{slug}.json"
    filepath = REPORTS_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Also save a markdown version
    md_path = filepath.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {result['query']}\n\n")
        f.write(f"**Model:** {result['model']}\n\n")
        f.write(f"**Time:** {result['elapsed_seconds']}s | {result['message_count']} messages\n\n")
        f.write(f"**Generated:** {result['timestamp']}\n\n")
        f.write("---\n\n")
        f.write(result["final_report"])
    
    return filepath, md_path


def watch_queue(queue_file: str = None):
    """Watch a JSONL task queue and process incoming research tasks."""
    if queue_file is None:
        queue_file = str(PROJECT_DIR / "research_queue.jsonl")
    
    print(f"🔍 Watching queue: {queue_file}")
    print(f"   Reports saved to: {REPORTS_DIR}")
    print(f"   Press Ctrl+C to stop\n")
    
    processed = set()
    
    while True:
        if not os.path.exists(queue_file):
            Path(queue_file).touch()
        
        with open(queue_file) as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or i in processed:
                continue
            
            try:
                task = json.loads(line)
                query = task.get("query", "")
                if not query:
                    continue
                
                priority = task.get("priority", "normal")
                print(f"\n{'='*60}")
                print(f"🔬 [{priority.upper()}] {query[:100]}...")
                
                result = run_research(
                    query=query,
                    max_iterations=task.get("max_iterations", 3),
                    search_api=task.get("search_api", "none"),
                )
                
                json_path, md_path = save_report(result)
                print(f"✅ Report saved: {md_path}")
                print(f"   Time: {result['elapsed_seconds']}s")
                processed.add(i)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                processed.add(i)
        
        time.sleep(10)  # poll interval


def main():
    parser = argparse.ArgumentParser(description="Nemotron Deep Research Agent")
    parser.add_argument("query", nargs="?", help="Research question")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--watch", action="store_true", help="Watch queue for tasks")
    parser.add_argument("--batch", help="Process JSONL batch file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    if args.watch:
        watch_queue()
        return
    
    if args.batch:
        print("Batch mode — reading tasks...")
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                task = json.loads(line)
                query = task.get("query", "")
                print(f"\n🔬 {query[:80]}...")
                result = run_research(query, max_iterations=task.get("max_iterations", 3))
                save_report(result)
                print(f"   ✅ {result['elapsed_seconds']}s")
        return
    
    if not args.query:
        parser.print_help()
        return
    
    # Single research
    result = run_research(args.query, max_iterations=args.iterations)
    json_path, md_path = save_report(result)
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"🔬 Research complete: {args.query}")
        print(f"   Model: {result['model']}")
        print(f"   Time: {result['elapsed_seconds']}s")
        print(f"   Report: {md_path}")
        print(f"{'='*60}\n")
        preview = result["final_report"][:800]
        print(preview)
        if len(result["final_report"]) > 800:
            print(f"\n... (full report at {md_path})")


if __name__ == "__main__":
    main()
