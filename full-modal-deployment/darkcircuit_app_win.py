import modal
from typing import List, Dict, Any, Generator, Optional
import time
import subprocess
import asyncio
import base64
import tempfile
import paramiko
import os

MINUTES = 60  # seconds

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "paramiko", "ollama", "langchain", "sseclient-py")
)

app = modal.App("DarkCircuit")

# Store SSH connection information
ssh_state = {
    "client": None,
    "channel": None,
    "connected": False,
    "error": None,
    "running": False
}


def setup_ssh_connection(host: str, port: int, username: str, password: Optional[str] = None,
                         key_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Establish an SSH connection to an external server
    """
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect with either password or key
        if key_path:
            if password:
                key = paramiko.RSAKey.from_private_key_file(key_path, password=password)
                client.connect(hostname=host, port=port, username=username, pkey=key, timeout=10)
            else:
                client.connect(hostname=host, port=port, username=username, key_filename=key_path, timeout=10)
        else:
            client.connect(hostname=host, port=port, username=username, password=password, timeout=10)

        # Store the client for later use
        ssh_state["client"] = client
        ssh_state["connected"] = True
        ssh_state["error"] = None

        return {
            "success": True,
            "message": f"SSH connection established to {host}."
        }

    except Exception as e:
        ssh_state["connected"] = False
        ssh_state["error"] = str(e)
        return {
            "success": False,
            "error": f"Failed to establish SSH connection: {str(e)}"
        }


def close_ssh_connection():
    """Close the SSH connection and clean up resources"""
    ssh_state["running"] = False

    if ssh_state["channel"]:
        try:
            ssh_state["channel"].close()
        except:
            pass
        ssh_state["channel"] = None

    if ssh_state["client"]:
        try:
            ssh_state["client"].close()
        except:
            pass
        ssh_state["client"] = None

    ssh_state["connected"] = False
    ssh_state["error"] = None


# FastAPI private backend server
@app.function(
    image=app_image.add_local_dir("frontend/dist", remote_path="/assets"),
    scaledown_window=15 * MINUTES
)
@modal.asgi_app()
def App():
    import fastapi
    from fastapi import FastAPI, Request, Depends, WebSocket, WebSocketDisconnect, File, Form, UploadFile
    import websockets
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    import httpx
    import paramiko
    import os
    import json
    import time
    from typing import Optional, List, Dict, Any
    from pydantic import BaseModel
    import uuid

    # Define API models
    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]

    class ModelPullRequest(BaseModel):
        model: str

    OllamaServer = modal.Cls.from_name("Ollama-Server", "OllamaServer")
    ollama_server = OllamaServer()

    fastapi_app = FastAPI()

    # CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # FastAPI WebSocket endpoint for terminal with proper PTY handling
    @fastapi_app.websocket("/ws/ssh-terminal")
    async def terminal_endpoint(websocket: WebSocket):
        await websocket.accept()

        try:
            # Check if we have an SSH connection
            if not ssh_state["connected"] or not ssh_state["client"]:
                await websocket.send_text("No SSH connection established. Please connect first.")
                await websocket.close()
                return

            # Get transport and open an interactive session
            transport = ssh_state["client"].get_transport()
            ssh_state["channel"] = channel = transport.open_session()

            # Request a pseudo-terminal
            term = os.environ.get('TERM', 'xterm')
            channel.get_pty(term, 80, 24)

            # Start shell
            channel.invoke_shell()

            # Mark as running
            ssh_state["running"] = True

            # Function to read from SSH and send to WebSocket
            async def ssh_to_ws():
                try:
                    while ssh_state["running"] and not channel.exit_status_ready():
                        if channel.recv_ready():
                            data = channel.recv(1024)
                            if data:
                                # Send raw binary data for proper terminal rendering
                                await websocket.send_bytes(data)
                            else:
                                break
                        else:
                            await asyncio.sleep(0.05)
                except Exception as e:
                    print(f"Error in ssh_to_ws: {e}")
                    ssh_state["error"] = str(e)
                finally:
                    ssh_state["running"] = False

            # Function to read from WebSocket and send to SSH
            async def ws_to_ssh():
                try:
                    while ssh_state["running"]:
                        try:
                            # Receive both text and binary messages
                            message = await websocket.receive()

                            # Check message type
                            if "text" in message:
                                data = message["text"].encode()
                            elif "bytes" in message:
                                data = message["bytes"]
                            else:
                                continue

                            # Handle special terminal commands
                            if data == b'\x03':  # Ctrl+C
                                channel.send(data)
                            elif data.startswith(b'\x1b['):  # Terminal escape sequences
                                channel.send(data)
                            else:
                                channel.send(data)

                        except asyncio.TimeoutError:
                            # Check if connection is still alive
                            if not ssh_state["running"] or channel.exit_status_ready():
                                break

                except Exception as e:
                    print(f"Error in ws_to_ssh: {e}")
                    ssh_state["error"] = str(e)
                finally:
                    ssh_state["running"] = False

            # Run both directions concurrently
            await asyncio.gather(
                ssh_to_ws(),
                ws_to_ssh()
            )

        except Exception as e:
            print(f"Error in terminal_endpoint: {e}")
            await websocket.send_text(f"Error: {str(e)}")

        finally:
            # Clean up SSH channel but keep the SSH connection
            if ssh_state["channel"]:
                try:
                    ssh_state["channel"].close()
                except:
                    pass
                ssh_state["channel"] = None

            ssh_state["running"] = False

            # Close WebSocket if still open
            try:
                await websocket.close()
            except:
                pass

    # Handle running commands on the remote server via SSH
    async def run_ssh_command(command: str) -> Dict[str, Any]:
        """Run a command on the SSH server and return the results"""
        if not ssh_state["connected"] or not ssh_state["client"]:
            return {
                "success": False,
                "output": "Not connected to SSH server.",
                "error": "Not connected"
            }

        try:
            stdin, stdout, stderr = ssh_state["client"].exec_command(command, timeout=30)
            exit_status = stdout.channel.recv_exit_status()

            output = stdout.read().decode('utf-8', errors='replace')
            error = stderr.read().decode('utf-8', errors='replace')

            return {
                "success": exit_status == 0,
                "output": output,
                "error": error,
                "exit_code": exit_status
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1
            }

    # API endpoint to SSH and set up the terminal
    @fastapi_app.post("/api/ssh/connect")
    async def connect_ssh(request: Request):
        try:
            data = await request.json()

            host = data.get("host", "")
            port = int(data.get("port", 22))
            username = data.get("username", "")
            password = data.get("password", "")
            key_path = data.get("key_path", "")

            # Close any existing connection
            close_ssh_connection()

            # Connect to SSH server
            result = setup_ssh_connection(host, port, username, password, key_path if key_path else None)

            return result

        except Exception as e:
            return {"success": False, "error": f"Error connecting to SSH: {str(e)}"}

    @fastapi_app.post("/api/ssh/disconnect")
    async def disconnect_ssh():
        """Disconnect from SSH server"""

        # Now close the SSH connection
        close_ssh_connection()
        return {"success": True, "message": "SSH connection closed"}


    @fastapi_app.get("/api/models")
    async def get_models():
        """Get available models from Ollama"""
        models = ollama_server.tags.remote()
        return {"models": models}

    @fastapi_app.post("/api/models/pull")
    async def pull_model(req: ModelPullRequest):
        """Pull a new model onto the Ollama server."""
        try:
            result = ollama_server.pull.remote(req.model)
            return {"success": True, "message": f"Model '{req.model}' pulled successfully."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @fastapi_app.delete("/api/models/{model_name}")
    async def delete_model(model_name: str):
        """Delete a model from the Ollama server."""
        try:
            result = ollama_server.delete.remote(model_name)
            return {"success": True, "message": f"Model '{model_name}' deleted successfully."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @fastapi_app.post("/api/chat/completions")
    async def chat_completions(request: ChatRequest):
        """
        REST API endpoint for chat completions with streaming.
        Compatible with OpenAI-style API for LangGraph integration.
        """
        # Convert to format expected by Ollama
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Use the async streaming method
        # Streaming response generator
        async def generate_stream():
            for chunk in ollama_server.chat.remote_gen(request.model, ollama_messages):
                # Format each chunk as an SSE event in OpenAI format
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk,
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Final chunk with finish_reason
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    # Static files
    fastapi_app.mount("/", StaticFiles(directory="/assets", html=True), name="frontend")

    return fastapi_app