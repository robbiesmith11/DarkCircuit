import asyncio
import paramiko
import os
import re
import json
import time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from pydantic import BaseModel
from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server
import sys

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatResult

from darkcircuit_agent_modular import Darkcircuit_Agent
from utils import get_path

# FastAPI backend server
app = FastAPI()

# CORS middleware
app.add_middleware(
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
    reasoner_prompt: Optional[str] = None
    responder_prompt: Optional[str] = None

class TerminalOutput(BaseModel):
    command_id: int
    output: str

# Initialize SSH state
ssh_state = {
    "client": None,
    "channel": None,
    "connected": False,
    "error": None,
    "running": False
}

# Store active agent instances
active_agents = {}
# Map WebSocket connections to their terminal output buffers
terminal_output_buffers = {"main": ""}
# Store active WebSocket connections for terminals
active_terminal_connections = set()
# Map to send terminal outputs to corresponding WebSockets
terminal_ws_clients = set()

# Global lock to serialize commands
command_lock = asyncio.Lock()

# Unique ready-marker to inject at each prompt
PROMPT_READY_MARKER = "\033]1337;CMD_READY\007"

def _strip_ansi_codes(text: any):
    """
    Strip ANSI escape sequences from a string

    Args:
        text (any): The text containing ANSI escape codes

    Returns:
        str: Clean text without ANSI codes
    """
    # This regex matches all ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text) if isinstance(text, str) else text

def _setup_ssh_connection(host: str, port: int, username: str, password: Optional[str] = None,
                         key_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Establish an SSH connection to an external server
    """
    global ssh_state
    
    # First, ensure any existing connection is properly closed
    _close_ssh_connection()

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
        ssh_state["client"] = client
        ssh_state["connected"] = True
        ssh_state["error"] = None

        return {
            "success": True,
            "message": f"SSH connection established to {host}."
        }

    except paramiko.ssh_exception.AuthenticationException as e:
        ssh_state["connected"] = False
        ssh_state["error"] = "Authentication failed. Please check your username, password or key."
        return {
            "success": False,
            "error": "Authentication failed. Please check your username, password or key."
        }
    except paramiko.ssh_exception.NoValidConnectionsError as e:
        ssh_state["connected"] = False
        ssh_state["error"] = f"Could not connect to {host}:{port}. Server may be down or unreachable."
        return {
            "success": False,
            "error": f"Could not connect to {host}:{port}. Server may be down or unreachable."
        }
    except Exception as e:
        ssh_state["connected"] = False
        ssh_state["error"] = str(e)
        return {
            "success": False,
            "error": f"Failed to establish SSH connection: {str(e)}"
        }

def _close_ssh_connection():
    """Close the SSH connection and clean up resources"""
    global ssh_state
    
    ssh_state["running"] = False

    if ssh_state["channel"]:
        try:
            ssh_state["channel"].close()
        except:
            pass
        ssh_state["channel"] = None

    if ssh_state["client"]:
        try:
            # Try graceful shutdown first
            transport = ssh_state["client"].get_transport()
            if transport and transport.is_active():
                transport.close()

            # Now close the client
            ssh_state["client"].close()
        except:
            pass
        ssh_state["client"] = None

    ssh_state["connected"] = False
    ssh_state["error"] = None


async def run_ssh_command(command: str, timeout: int = 1200) -> Dict[str, Any]:
    global terminal_ws_clients, terminal_output_buffers

    if not terminal_ws_clients:
        return {"success": False, "output": "", "error": "No terminal", "exit_code": -1}

    websocket = next(iter(terminal_ws_clients))

    async with command_lock:
        # Clear buffer and send the command
        terminal_output_buffers["main"] = ""
        # Add a marker to identify agent commands
        marked_command = f"__AGENT_COMMAND__:{command}\n"
        await websocket.send_text(marked_command)

        start = time.time()
        last_output = ""
        output_stable_count = 0

        while time.time() - start < timeout:
            await asyncio.sleep(0.1)
            raw = terminal_output_buffers["main"]

            # Detect prompt
            if raw == last_output:
                output_stable_count += 1
                # If output hasn't changed for ~1 second (10 * 0.1s), check for CLI prompts
                if output_stable_count >= 10:
                    cleaned = _strip_ansi_codes(raw)

                    # Look for CLI prompt patterns at the end of the output
                    cli_patterns = [
                        # Shell prompts
                        r'└──╼ [★]$', # Standard HTB prompt
                        r'smb: \\[^\\]*> $',  # SMB client
                        r'ftp> $',  # FTP client
                        r'mysql> $',  # MySQL client
                        r'sqlite> $',  # SQLite client
                        r'irb\([^)]*\)> $',  # Interactive Ruby
                        r'>>> $',  # Python REPL
                        r'\$ $',  # Bash prompt
                        r'# $',  # Root prompt
                        r'> $',  # Generic prompt

                        # Security tools & debuggers
                        r'msf.*?> $',  # Metasploit Framework
                        r'\(gdb\) $',  # GDB debugger
                        r'\[0x[0-9a-f]+\]> $',  # Radare2
                        r'sqlmap.*?> $',  # SQLMap interactive
                        r'bettercap.*?> $',  # Bettercap

                        # Authentication prompts
                        r'[Ll]ogin: $',  # Login prompt
                        r'[Pp]assword: $',  # Password prompt
                        r'[Uu]sername: $',  # Username prompt
                        r'[Pp]assphrase.*?: $',  # Passphrase prompt
                        r'Password for \[WORKGROUP\\\]:', # RPC Client Login

                        # Other interactive interfaces
                        r'Command> $',  # Generic command prompt
                        r'[Ss]hell> $',  # Generic shell prompt
                        r'postgres=# $',  # PostgreSQL
                        r'.*?=# $',  # Database CLI
                        r'.*?=> $',  # Alternative database prompt
                    ]

                    # Get the last line of output (where prompt likely is)
                    lines = cleaned.splitlines()
                    if lines:
                        last_line = lines[-1]

                        for pattern in cli_patterns:
                            if re.search(pattern, last_line):
                                print(f"Detected interactive CLI prompt: {pattern} in line: {last_line}")
                                return {"success": True, "output": cleaned, "error": "", "exit_code": 0}
            else:
                # Reset the counter when output changes
                output_stable_count = 0
                last_output = raw

        # Timeout: return whatever we have
        final = _strip_ansi_codes(terminal_output_buffers["main"])
        return {"success": False, "output": final, "error": "Timed out", "exit_code": -1}


# API endpoint to SSH and set up the terminal
@app.post("/api/ssh/connect")
async def connect_ssh(request: Request):
    try:
        data = await request.json()

        host = data.get("host", "")
        port = int(data.get("port", 22))
        username = data.get("username", "")
        password = data.get("password", "")
        key_path = data.get("key_path", "")

        # Connect to SSH server
        result = _setup_ssh_connection(host, port, username, password, key_path if key_path else None)

        return result

    except Exception as e:
        return {"success": False, "error": f"Error connecting to SSH: {str(e)}"}

@app.post("/api/ssh/disconnect")
async def disconnect_ssh():
    """Disconnect from SSH server"""

    # Now close the SSH connection
    _close_ssh_connection()
    return {"success": True, "message": "SSH connection closed"}

@app.post("/api/terminal/output")
async def submit_terminal_output(output_data: TerminalOutput):
    """
    This endpoint is kept for compatibility with the frontend,
    but we're not using it for command output collection as we now
    execute commands directly on the backend.
    """
    return {"success": True}

# REST API endpoint for chat completions with streaming
@app.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    REST API endpoint for chat completions with streaming.
    """
    # Convert to format expected by the agent
    prompt = request.messages[-1].content  # Take the latest user message as prompt

    # Create the agent with direct SSH command execution capability
    agent = Darkcircuit_Agent(
        model_name=request.model,
        reasoning_prompt=request.reasoner_prompt,
        response_prompt=request.responder_prompt,
        ssh_command_runner=run_ssh_command  # Pass our SSH command runner
    )

    # Store the agent for later reference
    agent_id = f"agent_{int(time.time())}"
    active_agents[agent_id] = agent

    async def generate_stream():
        try:
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

                # For UI terminal command events - these are now mostly for UI feedback
                elif event_type == "ui_terminal_command":
                    # We don't need to wait for the frontend to execute the command
                    # as we're now executing it directly, but we still want to inform
                    # the frontend about the command for display purposes
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
        finally:
            # Clean up the agent after the conversation is complete
            if agent_id in active_agents:
                del active_agents[agent_id]

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

# WebSocket endpoint for terminal
@app.websocket("/ws/ssh-terminal")
async def terminal_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Add to the list of websockets to receive SSH command outputs
    terminal_ws_clients.add(websocket)

    try:
        # Check if we have an SSH connection
        if not ssh_state["connected"] or not ssh_state["client"]:
            await websocket.send_text("No SSH connection established. Please connect first.")
            await websocket.close()
            return

        # Get transport and open an interactive session
        try:
            transport = ssh_state["client"].get_transport()
            if transport is None or not transport.is_active():
                ssh_state["connected"] = False
                await websocket.send_text("SSH transport is no longer active. Please reconnect.")
                await websocket.close()
                return

            transport.set_keepalive(30)  # 30s transport-layer keepalives
            ssh_state["channel"] = channel = transport.open_session()
        except Exception as e:
            ssh_state["connected"] = False
            ssh_state["error"] = str(e)
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
            ssh_state["connected"] = False
            ssh_state["error"] = str(e)
            await websocket.send_text(f"Failed to initialize terminal: {str(e)}")
            await websocket.close()
            return

        # Mark as running
        ssh_state["running"] = True

        # Function to read from SSH and send to WebSocket
        async def ssh_to_ws():
            try:
                while ssh_state["running"] and not channel.exit_status_ready():
                    # When receiving output from SSH, update the global buffer:
                    if channel.recv_ready():
                        data = channel.recv(1024)
                        if data:
                            # Send raw binary data for proper terminal rendering
                            await websocket.send_bytes(data)

                            # Also add to the global output buffer
                            try:
                                decoded_data = data.decode('utf-8', errors='replace')
                                terminal_output_buffers["main"] += decoded_data

                                # Limit buffer size to prevent memory issues
                                if len(terminal_output_buffers["main"]) > 10000:
                                    terminal_output_buffers["main"] = terminal_output_buffers["main"][-10000:]
                            except Exception as e:
                                print(f"Error decoding terminal data: {e}")
                    else:
                        # Short sleep to prevent CPU spinning
                        await asyncio.sleep(0.05)

            except Exception as e:
                print(f"Error in ssh_to_ws: {e}")
                ssh_state["error"] = str(e)
                # Try to inform the client if possible
                try:
                    await websocket.send_text(f"\r\n\x1b[1;31mError: {str(e)}\x1b[0m")
                except:
                    pass
            finally:
                ssh_state["running"] = False

        # Function to read from WebSocket and send to SSH
        async def ws_to_ssh():
            try:
                while ssh_state["running"]:
                    try:
                        # Receive both text and binary messages with a timeout
                        message = await asyncio.wait_for(
                            websocket.receive(),
                            timeout=1.0  # 1 second timeout to check if connection is still running
                        )

                        # Check message type
                        if "text" in message:
                            data = message["text"]
                            # Regular text command
                            data = data.encode()
                            channel.send(data)

                        elif "bytes" in message:
                            data = message["bytes"]
                            channel.send(data)
                        else:
                            continue

                    except asyncio.TimeoutError:
                        # Just a timeout, loop and try again if we're still supposed to be running
                        if not ssh_state["running"] or channel.exit_status_ready():
                            break
                    except Exception as e:
                        # Handle any WebSocket closure or disconnection
                        print(f"WebSocket receive error: {e}")
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
        try:
            await websocket.send_text(f"Error: {str(e)}")
        except:
            pass


    finally:
        # Remove from active connections and clients list
        terminal_ws_clients.discard(websocket)


static_path = get_path('frontend/dist')

if not os.path.isdir(static_path):
    raise RuntimeError(f"❌ Static path not found: {static_path}")

app.mount("/", StaticFiles(directory=static_path, html=True), name="frontend")



if __name__ == "__main__":

    async def run_uvicorn():
        config = Config(app=app, host="127.0.0.1", port=8000, loop="asyncio", lifespan="on")
        server = Server(config)
        await server.serve()

    try:
        asyncio.run(run_uvicorn())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("🛑 Backend server shutdown cleanly.")
