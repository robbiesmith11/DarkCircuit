import modal
from typing import List, Dict, Any, Generator

GPU = "T4"
N_GPU = 1
MINUTES = 60  # seconds
API_KEY_SECRET = modal.Secret.from_name("ollama-secret")
OLLAMA_PORT = 11434

ollama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "systemctl", "lshw")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh", gpu=GPU
    )
    .pip_install("ollama", "httpx", "loguru")
)

MODEL_VOLUME = modal.Volume.from_name("ollama-models", create_if_missing=True)

app = modal.App("Ollama-Server")

with ollama_image.imports():
    import ollama
    from ollama import AsyncClient
    import subprocess
    import httpx, time


# Ollama private server
@app.cls(
    image=ollama_image,
    gpu=f"{GPU}:{N_GPU}",
    volumes={"/root/.ollama/models": MODEL_VOLUME},
    scaledown_window=1 * MINUTES  # 1 minutes
)
class OllamaServer:
    @modal.enter()
    def run_ollama(self):
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait_for_ollama()

    def wait_for_ollama(self, timeout: int = 60):
        start_time = time.time()
        while True:
            try:
                response = httpx.get("http://localhost:11434/api/version")
                if response.status_code == 200:
                    break
            except Exception:
                if time.time() - start_time > timeout:
                    raise RuntimeError("Ollama failed to start in time.")
                time.sleep(2)

    @modal.method()
    async def chat_async(self, model: str, messages: list):
        """
        Async chat with the Ollama model and yield response chunks.
        Uses the AsyncClient from the ollama library.
        """
        client = AsyncClient(host="http://localhost:11434")
        async for chunk in await client.chat(
                model=model,
                messages=messages,
                stream=True
        ):
            yield chunk["message"]["content"]

    @modal.method()
    def chat(self, model: str, messages: list) -> Generator[str, None, None]:
        """
        Synchronous chat with the Ollama model and yield response chunks.
        """
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    @modal.method()
    def tags(self) -> List[Dict[str, Any]]:
        """Get available models from Ollama"""
        return ollama.list()["models"]

    @modal.method()
    def pull(self, model: str):
        """Pull a new model onto the Ollama server."""
        return ollama.pull(model)

    @modal.method()
    def delete(self, model: str):
        """Delete a model from the Ollama server."""
        return ollama.delete(model)