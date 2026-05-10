"""
Nemotron 3 33B Ollama deployment on Modal — MindExpander Research Backend.
Deploys Ollama 0.23.x with nemotron3:33b (Q4_K_M, multimodal) on A100-40GB.

Usage:
  modal deploy modal/nemotron_ollama_deploy.py          # deploy service
  modal run modal/nemotron_ollama_deploy.py::pull_model   # pull the model
  modal run modal/nemotron_ollama_deploy.py::health       # verify readiness
"""
import subprocess, time, os
import modal

# ── Image: Ollama 0.23.x + deps ────────────────────────────────────────────
OLLAMA_VERSION = "0.23.1"  # latest as of May 2026
MODEL_NAME = "nemotron3:33b"
GPU_TYPE = "A100-40GB"     # 28GB model needs ~32GB+ VRAM

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "systemctl", "zstd", "ca-certificates")
    .run_commands(
        # Install latest Ollama
        f"curl -fsSL https://ollama.com/install.sh | sh",
        f"ollama --version",
    )
    .pip_install("httpx", "loguru", "requests")
    .env({
        "OLLAMA_HOST": "0.0.0.0:11434",
        "OLLAMA_MODELS": "/models",
        "OLLAMA_NUM_PARALLEL": "2",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "30m",  # keep model in memory for 30 min
    })
)

# Persistent volume for model weights
volume = modal.Volume.from_name("ollama-nemotron-models", create_if_missing=True)
app = modal.App(name="mindexpander-nemotron-ollama", image=image)


def wait_for_ollama(timeout: int = 60, interval: int = 3) -> None:
    """Wait for Ollama server to be ready."""
    import httpx
    start = time.time()
    while True:
        try:
            resp = httpx.get("http://localhost:11434/api/version", timeout=5)
            if resp.status_code == 200:
                version = resp.json().get("version", "unknown")
                print(f"Ollama {version} ready ({time.time() - start:.0f}s)")
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Ollama failed to start within {timeout}s")
        print(f"Waiting for Ollama... ({time.time() - start:.0f}s)")
        time.sleep(interval)


@app.cls(
    scaledown_window=600,  # 10 min idle before scale down
    volumes={"/models": volume},
    memory=8192,           # 8GB RAM for 33B model
    gpu=GPU_TYPE,
    timeout=900,           # 15 min for long generations
)
@modal.concurrent(max_inputs=1)
class NemotronService:
    """Nemotron 3 33B via Ollama on Modal."""

    @modal.enter()
    def start_ollama(self):
        """Start Ollama server."""
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_ollama()
        
        # Check if model is already pulled
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if MODEL_NAME not in result.stdout:
            print(f"Model {MODEL_NAME} not found — pulling now...")
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
            volume.commit()
            print(f"✅ Model {MODEL_NAME} pulled and cached")
        else:
            print(f"✅ Model {MODEL_NAME} already cached")

    @modal.web_server(11434)
    def server(self):
        """Expose Ollama API."""
        pass

    @modal.method()
    def health(self) -> dict:
        """Check service health + model status."""
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "model_loaded": MODEL_NAME in models,
            "gpu": GPU_TYPE,
            "models_available": models,
        }

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> dict:
        """Generate text via Ollama API."""
        import httpx, json
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": 8192,
            }
        }
        start = time.time()
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=300
        )
        result = resp.json()
        elapsed = time.time() - start
        
        tok_count = result.get("eval_count", 0)
        tok_per_sec = tok_count / elapsed if elapsed > 0 else 0
        
        return {
            "response": result.get("response", ""),
            "tokens": tok_count,
            "tokens_per_second": round(tok_per_sec, 1),
            "elapsed_seconds": round(elapsed, 2),
            "model": MODEL_NAME,
        }


@app.local_entrypoint()
def pull_model():
    """One-time: pull nemotron3:33b into the persistent volume."""
    service = NemotronService()
    result = service.health.remote()
    print(f"Health: {result}")
    print(f"Model loaded: {result['model_loaded']}")


@app.local_entrypoint()
def health():
    """Check service health."""
    service = NemotronService()
    result = service.health.remote()
    print(f"Status: {result['status']}")
    print(f"Model: {result['model']}")
    print(f"Loaded: {result['model_loaded']}")
    print(f"GPU: {result['gpu']}")
    return result


@app.local_entrypoint()
def test_generate():
    """Smoke test: generate a response."""
    service = NemotronService()
    result = service.generate.remote(
        prompt="Explain what LangGraph is in one paragraph.",
        max_tokens=256,
    )
    print(f"Response: {result['response'][:500]}")
    print(f"Tokens: {result['tokens']} ({result['tokens_per_second']} tok/s)")
    print(f"Time: {result['elapsed_seconds']}s")
