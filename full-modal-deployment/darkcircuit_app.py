import modal

MINUTES = 60  # seconds

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "paramiko", "ollama", "langchain", "sseclient-py", "langchain_community", "langgraph", "langchain_core", "langchain_openai", "duckduckgo-search==7.5.5", "langchain_text_splitters", "pypdf", "fastembed", "faiss-cpu")
    .add_local_dir("docs", remote_path="/docs")
    .add_local_python_source("darkcircuit_agent")
    .add_local_python_source("Rag_tool")
)

with app_image.imports():
    import asyncio
    import paramiko
    import os
    import json
    import time
    from typing import Optional, List, Dict, Any
    from pydantic import BaseModel
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    from langchain_core.runnables import Runnable

    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import Tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_core.outputs import ChatGeneration, ChatResult

    from darkcircuit_agent import Darkcircuit_Agent


app = modal.App("DarkCircuit")

ssh_volume = modal.Volume.from_name("ssh_data", create_if_missing=True)

# FastAPI private backend server
@app.cls(
    image=app_image.add_local_dir("frontend/dist", remote_path="/assets"),
    timeout=30*MINUTES,
    scaledown_window=15*MINUTES,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={"/ssh_data": ssh_volume}
)
class App:
    def __init__(self):

        # Store SSH connection information
        self.ssh_state = {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }

        OllamaServer = modal.Cls.from_name("Ollama-Server", "OllamaServer")
        self.ollama_server = OllamaServer()

        self.fastapi_app = FastAPI()

        # CORS middleware
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Define API models
        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            model: str
            messages: List[ChatMessage]

        class ModelPullRequest(BaseModel):
            model: str


        # API endpoint to SSH and set up the terminal
        @self.fastapi_app.post("/api/ssh/connect")
        async def connect_ssh(request: Request):
            try:
                data = await request.json()

                host = data.get("host", "")
                port = int(data.get("port", 22))
                username = data.get("username", "")
                password = data.get("password", "")
                key_path = data.get("key_path", "")

                # Connect to SSH server
                result = self._setup_ssh_connection(host, port, username, password, key_path if key_path else None)


                return result

            except Exception as e:
                return {"success": False, "error": f"Error connecting to SSH: {str(e)}"}

        @self.fastapi_app.post("/api/ssh/disconnect")
        async def disconnect_ssh():
            """Disconnect from SSH server"""

            # Now close the SSH connection
            self._close_ssh_connection()
            return {"success": True, "message": "SSH connection closed"}

        @self.fastapi_app.get("/api/models")
        async def get_models():
            """Get available models from Ollama"""
            models = self.ollama_server.tags.remote()
            return {"models": models}

        @self.fastapi_app.post("/api/models/pull")
        async def pull_model(req: ModelPullRequest):
            """Pull a new model onto the Ollama server."""
            try:
                result = self.ollama_server.pull.remote(req.model)
                return {"success": True, "message": f"Model '{req.model}' pulled successfully."}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @self.fastapi_app.delete("/api/models/{model_name}")
        async def delete_model(model_name: str):
            """Delete a model from the Ollama server."""
            try:
                result = self.ollama_server.delete.remote(model_name)
                return {"success": True, "message": f"Model '{model_name}' deleted successfully."}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @self.fastapi_app.post("/api/chat/completions")
        async def chat_completions(request: ChatRequest):
            """
            REST API endpoint for chat completions with streaming.
            Compatible with OpenAI-style API for LangGraph integration.
            """
            # Convert to format expected by Ollama
            prompt = request.messages[-1].content  # Take the latest user message as prompt

            # Load connection parameters from the volume
            with open("/ssh_data/connection_params.json", "r") as f:
                conn_params = json.load(f)

            print(f"main app: {conn_params}")

            # Run the LangGraph agent with the custom chat model
            agent = Darkcircuit_Agent()

            async def generate_stream():
                async for event in agent.run_agent_streaming(prompt):
                    event_type = event.get("type", "unknown")

                    # For token events (chat content)
                    if event_type == "token":
                        token = event.get("value", "")
                        data = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token, "role": "assistant"},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    # For thinking, tool_call, and tool_result events
                    elif event_type in ["thinking", "tool_call", "tool_result"]:
                        # Forward these events directly for the debug panel
                        yield f"data: {json.dumps(event)}\n\n"

                # Final stop chunk
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

        # FastAPI WebSocket endpoint for terminal
        @self.fastapi_app.websocket("/ws/ssh-terminal")
        async def terminal_endpoint(websocket: WebSocket):
            await websocket.accept()

            try:
                # Check if we have an SSH connection
                if not self.ssh_state["connected"] or not self.ssh_state["client"]:
                    await websocket.send_text("No SSH connection established. Please connect first.")
                    await websocket.close()
                    return

                # Get transport and open an interactive session
                try:
                    transport = self.ssh_state["client"].get_transport()
                    if transport is None or not transport.is_active():
                        self.ssh_state["connected"] = False
                        await websocket.send_text("SSH transport is no longer active. Please reconnect.")
                        await websocket.close()
                        return

                    self.ssh_state["channel"] = channel = transport.open_session()
                except Exception as e:
                    self.ssh_state["connected"] = False
                    self.ssh_state["error"] = str(e)
                    await websocket.send_text(f"Failed to establish SSH channel: {str(e)}")
                    await websocket.close()
                    return

                # Request a interactive shell
                try:
                    term = os.environ.get('TERM', 'xterm')
                    channel.get_pty(term, 80, 24)

                    # Start shell
                    channel.invoke_shell()
                except Exception as e:
                    self.ssh_state["connected"] = False
                    self.ssh_state["error"] = str(e)
                    await websocket.send_text(f"Failed to initialize terminal: {str(e)}")
                    await websocket.close()
                    return

                # Mark as running
                self.ssh_state["running"] = True

                # Function to read from SSH and send to WebSocket
                async def ssh_to_ws():
                    try:
                        while self.ssh_state["running"] and not channel.exit_status_ready():
                            if channel.recv_ready():
                                data = channel.recv(1024)
                                if data:
                                    # Send raw binary data for proper terminal rendering
                                    await websocket.send_bytes(data)
                                else:
                                    break
                            else:
                                # Short sleep to prevent CPU spinning
                                await asyncio.sleep(0.05)

                                # Check if channel is still open but not sending data
                                if not channel.exit_status_ready() and not channel.recv_ready():
                                    try:
                                        # Try sending a null byte to keep the connection alive
                                        # This acts like a keepalive packet
                                        if time.time() % 30 < 0.1:  # Approximately every 30 seconds
                                            channel.send('\0')
                                    except:
                                        # If we can't send, channel is likely dead
                                        break
                    except Exception as e:
                        print(f"Error in ssh_to_ws: {e}")
                        self.ssh_state["error"] = str(e)
                        # Try to inform the client if possible
                        try:
                            await websocket.send_text(f"\r\n\x1b[1;31mError: {str(e)}\x1b[0m")
                        except:
                            pass
                    finally:
                        self.ssh_state["running"] = False

                # Function to read from WebSocket and send to SSH
                async def ws_to_ssh():
                    try:
                        while self.ssh_state["running"]:
                            try:
                                # Receive both text and binary messages with a timeout
                                message = await asyncio.wait_for(
                                    websocket.receive(),
                                    timeout=1.0  # 1 second timeout to check if connection is still running
                                )

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
                                # Just a timeout, loop and try again if we're still supposed to be running
                                if not self.ssh_state["running"] or channel.exit_status_ready():
                                    break
                            except Exception as e:
                                # Handle any WebSocket closure or disconnection
                                print(f"WebSocket receive error: {e}")
                                break

                    except Exception as e:
                        print(f"Error in ws_to_ssh: {e}")
                        self.ssh_state["error"] = str(e)
                    finally:
                        self.ssh_state["running"] = False

                # Run both directions concurrently
                await asyncio.gather(
                    ssh_to_ws(),
                    ws_to_ssh()
                )

            except Exception as e:
                print(f"Error in terminal_endpoint: {e}")
                try:
                    await websocket.send_text(f"Error: {str(e)}")
                except:
                    pass

            finally:
                # Clean up SSH channel but keep the SSH connection
                if self.ssh_state["channel"]:
                    try:
                        self.ssh_state["channel"].close()
                    except:
                        pass
                    self.ssh_state["channel"] = None

                self.ssh_state["running"] = False

                # Close WebSocket if still open
                try:
                    await websocket.close()
                except:
                    pass

    def _setup_ssh_connection(self, host: str, port: int, username: str, password: Optional[str] = None,
                             key_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Establish an SSH connection to an external server
        """
        # First, ensure any existing connection is properly closed
        self._close_ssh_connection()

        try:
            # Create SSH client
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Configure connection timeout and keep-alive
            connect_kwargs = {
                'hostname': host,
                'port': port,
                'username': username,
                'timeout': 10,
                'banner_timeout': 15,
                'allow_agent': False,
                'look_for_keys': False
            }

            # Add either password or key authentication
            if key_path:
                if password:
                    key = paramiko.RSAKey.from_private_key_file(key_path, password=password)
                    connect_kwargs['pkey'] = key
                else:
                    connect_kwargs['key_filename'] = key_path
            else:
                connect_kwargs['password'] = password

            # Connect to the SSH server
            client.connect(**connect_kwargs)

            # Enable keep-alive packets
            transport = client.get_transport()
            if transport:
                transport.set_keepalive(30)  # Send keep-alive every 30 seconds

            # Verify connection with a test command
            try:
                stdin, stdout, stderr = client.exec_command("echo connected", timeout=5)
                result = stdout.read().decode('utf-8', errors='replace').strip()
                if result != "connected":
                    raise Exception("Test command did not return expected output")
            except Exception as e:
                # If test fails, close connection and raise error
                client.close()
                raise Exception(f"Connection test failed: {str(e)}")

            # Store the client for later use
            self.ssh_state["client"] = client
            self.ssh_state["connected"] = True
            self.ssh_state["error"] = None

            # After successful connection, save the connection parameters to volume
            if client:
                connection_params = {
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "key_path": key_path
                }

                # Save to a file in the volume
                with open("/ssh_data/connection_params.json", "w") as f:
                    json.dump(connection_params, f)

            return {
                "success": True,
                "message": f"SSH connection established to {host}."
            }

        except paramiko.ssh_exception.AuthenticationException as e:
            self.ssh_state["connected"] = False
            self.ssh_state["error"] = "Authentication failed. Please check your username, password or key."
            return {
                "success": False,
                "error": "Authentication failed. Please check your username, password or key."
            }
        except paramiko.ssh_exception.NoValidConnectionsError as e:
            self.ssh_state["connected"] = False
            self.ssh_state["error"] = f"Could not connect to {host}:{port}. Server may be down or unreachable."
            return {
                "success": False,
                "error": f"Could not connect to {host}:{port}. Server may be down or unreachable."
            }
        except Exception as e:
            self.ssh_state["connected"] = False
            self.ssh_state["error"] = str(e)
            return {
                "success": False,
                "error": f"Failed to establish SSH connection: {str(e)}"
            }

    def _close_ssh_connection(self):
        """Close the SSH connection and clean up resources"""
        self.ssh_state["running"] = False

        if self.ssh_state["channel"]:
            try:
                self.ssh_state["channel"].close()
            except:
                pass
            self.ssh_state["channel"] = None

        if self.ssh_state["client"]:
            try:
                # Try graceful shutdown first
                transport = self.ssh_state["client"].get_transport()
                if transport and transport.is_active():
                    transport.close()

                # Now close the client
                self.ssh_state["client"].close()
            except:
                pass
            self.ssh_state["client"] = None

        self.ssh_state["connected"] = False
        self.ssh_state["error"] = None


    async def _run_ssh_command(self, command: str) -> Dict[str, Any]:
        """Run a command on the SSH server and return the results"""
        if not self.ssh_state["connected"] or not self.ssh_state["client"]:
            return {
                "success": False,
                "output": "Not connected to SSH server.",
                "error": "Not connected",
                "exit_code": -1
            }

        try:
            stdin, stdout, stderr = self.ssh_state["client"].exec_command(command, timeout=30)
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


    @modal.asgi_app()
    def serve(self):
        # Static files
        self.fastapi_app.mount("/", StaticFiles(directory="/assets", html=True), name="frontend")

        return self.fastapi_app


    class ModalOllamaChatModel(BaseChatModel, Runnable):
        model: str

        def __init__(self, ollama_server, model: str):
            object.__setattr__(self, "ollama_server", ollama_server)
            object.__setattr__(self, "model", model)

        def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
        ) -> ChatResult:
            ollama_messages = [{"role": self._convert_role(msg.type), "content": msg.content} for msg in messages]
            response = ""
            for chunk in self.ollama_server.chat.remote_gen(self.model, ollama_messages):
                response += chunk

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

        def _convert_role(self, msg_type: str) -> str:
            if msg_type == "human":
                return "user"
            elif msg_type == "ai":
                return "assistant"
            elif msg_type == "system":
                return "system"
            elif msg_type == "tool":
                return "tool"
            else:
                raise ValueError(f"Unsupported message type: {msg_type}")

        def invoke(self, input: List[BaseMessage], **kwargs: Any) -> AIMessage:
            return self._generate(input).generations[0].message

        async def ainvoke(self, input: List[BaseMessage], **kwargs: Any) -> AIMessage:
            return self._generate(input).generations[0].message

        def bind_tools(self, tools: List[Tool]):
            tool_schema = [convert_to_openai_tool(t) for t in tools]
            return self  # You can optionally store `tool_schema` and use it in `_generate` if your Ollama model supports tools.

        @property
        def _llm_type(self) -> str:
            return "modal_ollama"