import modal
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import subprocess
import os
import time


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
    .pip_install("fastapi[standard]", "ollama", "httpx", "loguru")
)

MODEL_VOLUME = modal.Volume.from_name("ollama-models", create_if_missing=True)

web_app = FastAPI()
app = modal.App("ollama")

auth_scheme = HTTPBearer()

def verify_api_key(token_cred):
    api_key = token_cred
    expected_key = os.environ.get("OLLAMA_API_KEY")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")



def start_server():
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()
    #subprocess.run(["ollama", "serve"])

def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready.

    :param timeout: Maximum time to wait in seconds
    :param interval: Time between checks in seconds
    :raises TimeoutError: If the service doesn't start within the timeout period
    """
    import httpx
    from loguru import logger

    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            logger.info(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


@web_app.post("/api/generate")
async def api_generate(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient
    from json import dumps

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")
    prompt = body.get("prompt")
    stream = body.get("stream", True)

    if not model or not prompt:
        raise HTTPException(status_code=400, detail="Missing model or prompt")

    if stream:
        # Define the streaming generator
        async def stream_response():
            async for part in await AsyncClient().generate(model=model, prompt=prompt, stream=True):
                yield part['response']

        # Return a streaming response
        return StreamingResponse(stream_response(), media_type="text/plain")
    else:
        # Define the async generator function
        async def generate_response():
            async for part in await AsyncClient().generate(model=model, prompt=prompt, stream=True):
                print(part['response'], end='', flush=True)

        # Await the function directly
        await generate_response()

@web_app.post("/api/chat")
async def api_chat(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    verify_api_key(token.credentials)

    from ollama import AsyncClient
    from json import dumps

    body = await request.json()
    model = body.get("name") or body.get("model")
    messages = body.get("messages")
    stream = body.get("stream", True)

    if not model or not messages:
        raise HTTPException(status_code=400, detail="Missing model or messages")

    client = AsyncClient()

    if stream:
        async def stream_response():
            async for part in await client.chat(model=model, messages=messages, stream=True):
                yield dumps(part.model_dump()) + '\n'  # Fix HERE!
        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
    else:
        response = await client.chat(model=model, messages=messages, stream=False)
        return response.model_dump()  # Fix HERE!


@web_app.post("/api/create")
async def api_create(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient
    from json import dumps

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")
    modelfile = body.get("modelfile")
    stream = body.get("stream", False)

    if not model or not modelfile:
        raise HTTPException(status_code=400, detail="Missing model or modelfile")

    if stream:
        async def stream_response():
            async for part in await AsyncClient().create(model=model, modelfile=modelfile, stream=True):
                yield dumps(part) + '\n'
        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
    else:
        response = await AsyncClient().create(model=model, modelfile=modelfile, stream=False)
        return response

@web_app.get("/api/tags", dependencies=[Depends(auth_scheme)])
async def api_tags(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    verify_api_key(token.credentials)

    from ollama import AsyncClient

    response = await AsyncClient().list()

    models_corrected = [
        {
            "name": model["model"],
            "model": model["model"],  # Add BOTH KEYS!
            "modified_at": model["modified_at"],
            "digest": model["digest"],
            "size": model["size"],
            "details": model.get("details", {})
        }
        for model in response.get("models", [])
    ]

    return {"models": models_corrected}


@web_app.post("/api/show")
async def api_show(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")

    if not model:
        raise HTTPException(status_code=400, detail="Missing model")

    response = await AsyncClient().show(model)
    return response

@web_app.post("/api/copy")
async def api_copy(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient

    verify_api_key(token.credentials)

    body = await request.json()
    source = body.get("source")
    destination = body.get("destination")

    if not source or not destination:
        raise HTTPException(status_code=400, detail="Missing source or destination")

    await AsyncClient().copy(source, destination)
    return {"status": "success"}

@web_app.delete("/api/delete")
async def api_delete(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")

    if not model:
        raise HTTPException(status_code=400, detail="Missing model")

    await AsyncClient().delete(model)
    return {"status": "success"}

@web_app.post("/api/pull")
async def api_pull(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient
    from json import dumps

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")
    stream = body.get("stream", False)

    if not model:
        raise HTTPException(status_code=400, detail="Missing model")

    if stream:
        async def stream_response():
            async for part in await AsyncClient().pull(model, stream=True):
                yield dumps(part) + '\n'
        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
    else:
        response = await AsyncClient().pull(model, stream=False)
        return response

@web_app.post("/api/push")
async def api_push(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient
    from json import dumps

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")
    stream = body.get("stream", False)

    if not model:
        raise HTTPException(status_code=400, detail="Missing model")

    if stream:
        async def stream_response():
            async for part in await AsyncClient().push(model, stream=True):
                yield dumps(part) + '\n'
        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
    else:
        response = await AsyncClient().push(model, stream=False)
        return response

@web_app.post("/api/embed")
async def api_embed(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient

    verify_api_key(token.credentials)

    body = await request.json()
    model = body.get("name") or body.get("model")
    input = body.get("input")  # Can be string or list of strings

    if not model or not input:
        raise HTTPException(status_code=400, detail="Missing model or input")

    response = await AsyncClient().embed(model, input)
    return response

@web_app.get("/api/ps")
async def api_ps(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    from ollama import AsyncClient

    verify_api_key(token.credentials)

    response = await AsyncClient().ps()
    return response

@web_app.get("/api/version")
async def api_version(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import httpx

    verify_api_key(token.credentials)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/version")
            response.raise_for_status()  # Raises an exception for 4xx/5xx responses
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.cls(
    image=ollama_image,
    gpu=f"{GPU}:{N_GPU}",
    secrets=[API_KEY_SECRET],
    volumes={"/root/.ollama/models": MODEL_VOLUME},
    allow_concurrent_inputs=50,
    scaledown_window=15 * MINUTES,  # 15 minutes
)
class App:
    @modal.enter()
    def enter(self):
        # Start Ollama server once at container startup
        subprocess.Popen([
            "ollama", "serve"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.wait_for_ollama()

    def wait_for_ollama(self, timeout: int = 60):
        import httpx, time
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

    @modal.asgi_app()
    def serve(self):
        start_server()
        return web_app
