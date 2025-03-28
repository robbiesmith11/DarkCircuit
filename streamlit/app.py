import subprocess
import os
import streamlit as st
import requests
import paramiko
import time
import threading
import json
from datetime import datetime
import html

# Environment variables
OLLAMA_API = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
KALI_SSH_HOST = os.environ.get("KALI_SSH_HOST", "offensive-docker")
KALI_SSH_PORT = 22
KALI_SSH_USER = os.environ.get("KALI_SSH_USER", "root")
KALI_SSH_PASS = os.environ.get("KALI_SSH_PASS", "root")
VPN_FILE_PATH = "/offensive/vpn/client.ovpn"

# Function to execute commands via SSH
def execute_command(command, terminal_history):
    timestamp = datetime.now().strftime('%H:%M:%S')
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(KALI_SSH_HOST, port=KALI_SSH_PORT, username=KALI_SSH_USER, password=KALI_SSH_PASS)
        stdin, stdout, stderr = ssh.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        encoded_output = html.escape(output)
        encoded_error = html.escape(error)

        if output:
            terminal_history.append(encoded_output)
        if error:
            terminal_history.append(f"[ERROR] {encoded_error}")
        if exit_status != 0:
            terminal_history.append(f"[EXIT STATUS] {exit_status}")
        ssh.close()
    except Exception as e:
        terminal_history.append(f"[{timestamp}] Error: {type(e).__name__}: {str(e)}")

# Main application
st.set_page_config(page_title="AI Hacking Lab", layout="wide")

# Custom CSS and JavaScript
st.markdown("""
<style>
    .terminal {
        background-color: #1E1E1E;
        color: #DCDCDC;
        font-family: monospace;
        padding: 10px;
        border-radius: 5px;
        height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 500px; /* Same height as terminal */
        overflow-y: auto; /* Enable scrolling */
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for VPN Connection and Container Controls
with st.sidebar:
    st.title("AI Hacking Lab")
    st.header("VPN Connection")

    # Handle VPN file upload and persist it in session_state
    uploaded_vpn_file = st.file_uploader("Upload your VPN (.ovpn) file", type=["ovpn"])

    if uploaded_vpn_file is not None:
        st.session_state["vpn_file"] = uploaded_vpn_file

    if "vpn_connected" not in st.session_state:
        st.session_state.vpn_connected = False

    unsupported_directives = ["data-ciphers-fallback", "data-ciphers"]

    if not st.session_state.vpn_connected:
        if st.button("Connect VPN"):
            vpn_file = st.session_state.get("vpn_file")
            if vpn_file:
                vpn_dir = "/offensive/vpn"
                vpn_path = os.path.join(vpn_dir, "client.ovpn")
                os.makedirs(vpn_dir, exist_ok=True)

                # Decode uploaded VPN file
                vpn_content = vpn_file.getvalue().decode("utf-8")

                # Programmatically comment out unsupported directives
                corrected_content = []
                for line in vpn_content.splitlines():
                    if any(directive in line for directive in unsupported_directives):
                        corrected_content.append(f"# {line}  # Commented out (unsupported directive)")
                    else:
                        corrected_content.append(line)

                # Write corrected content to file
                with open(vpn_path, "w") as f:
                    f.write("\n".join(corrected_content))

                # Connect to VPN via SSH
                connect_cmd = f"openvpn --config {vpn_path} --daemon"
                execute_command(
                    connect_cmd,
                    st.session_state.terminal_history
                )

                st.session_state.vpn_connected = True
                st.rerun()
            else:
                st.warning("Please upload a VPN file first.")
        st.info("No VPN Connected!")
    else:
        if st.button("Disconnect VPN"):
            disconnect_cmd = "pkill openvpn"
            st.session_state.terminal_history[-1] += disconnect_cmd
            execute_command(
                disconnect_cmd,
                st.session_state.terminal_history
            )
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")
            st.session_state.vpn_connected = False

            time.sleep(2)

            # Verify disconnection immediately
            verify_disconnect_cmd = "pgrep openvpn || echo 'VPN disconnected successfully!'"
            st.session_state.terminal_history[-1] += verify_disconnect_cmd
            execute_command(
                verify_disconnect_cmd,
                st.session_state.terminal_history
            )
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")
            st.rerun()
        st.success("VPN connected successfully!")

    st.header("Container Services")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Apache", use_container_width=True):
            try:
                subprocess.run("docker exec offensive-docker service apache2 start", shell=True, check=True)
                st.success("Apache started")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    with col2:
        if st.button("Start Squid", use_container_width=True):
            try:
                subprocess.run("docker exec offensive-docker service squid start", shell=True, check=True)
                st.success("Squid started")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    with st.expander("About Offensive Docker"):
        st.markdown("""
        Offensive Docker includes numerous pentesting tools organized in categories:
        - **Port scanning**: nmap, masscan, naabu
        - **Recon**: Amass, GoBuster, Sublist3r, etc.
        - **Web Scanning**: whatweb, wafw00z, nikto, etc.
        - **OWASP**: sqlmap, XSStrike, jwt_tool, etc.
        - **Wordlists**: SecList, dirb, wfuzz, rockyou

        Common utilities:
        - `apacheUp` - Start Apache web server
        - `squidUp` - Start Squid proxy server

        Most tools are installed in the `/tools` directory.
        **Important**: Save all your work in the `/offensive` directory to persist data!
        """)

# Fetch available models only once
if "available_models" not in st.session_state:
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags")
        st.session_state.available_models = [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        st.session_state.available_models = ["No models available"]
        st.warning(f"Failed to fetch models: {str(e)}")

# Main content area
col1, col2 = st.columns(2)

# LLM Chat Interface (Left Column)
with col1:
    st.header("AI Assistant")
    model_col, clear_col = st.columns([3, 1])
    with model_col:
        st.selectbox("Select Model", options=st.session_state.available_models, key="selected_model")
    with clear_col:
        if st.button("Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = [{
                "role": "system",
                "content": "I'm your AI assistant for security and penetration testing."
            }]
            st.rerun()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "system",
            "content": "I'm your AI assistant for security and penetration testing."
        }]

    # Render existing chat history
    chat_html = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_html += f"<div style='text-align: right; margin-bottom: 10px;'>👤 <b>User:</b> {message['content']}</div>"
        elif message["role"] == "assistant":
            chat_html += f"<div style='text-align: left; margin-bottom: 10px;'>🤖 <b>Assistant:</b> {message['content']}</div>"
    st.markdown(f'<div class="chat-container auto-scroll">{chat_html}</div>', unsafe_allow_html=True)

    # Handle new prompt with streaming
    prompt = st.chat_input("Ask me about security tools, techniques or concepts...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        if st.session_state.selected_model != "No models available":
            try:
                response = requests.post(
                    f"{OLLAMA_API}/api/generate",
                    json={"model": st.session_state.selected_model, "prompt": prompt, "stream": True},
                    stream=True
                )
                full_response = ""
                response_placeholder = st.empty()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("response", "")
                        full_response += chunk
                        response_placeholder.markdown(f"🤖 **Assistant:** {full_response}")
                        time.sleep(0.05)  # Small delay for UI update
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "No model available. Check Ollama service."
            })
        st.rerun()

# Terminal Interface (Right Column)
with col2:
    # Initialize terminal history
    if "terminal_history" not in st.session_state:
        st.session_state.terminal_history = ["$ "]

    # Terminal section
    st.header("Terminal")

    # Display terminal history with styling
    terminal_content = "\n".join(st.session_state.terminal_history)
    terminal_display = st.markdown(f'<div class="terminal auto-scroll">{terminal_content}</div>', unsafe_allow_html=True)

    # Chat input for terminal commands
    command = st.chat_input("Enter command...")

    if command:
        if command.lower() == "clear":
            # Clear the terminal history
            st.session_state.terminal_history = ["$ "]
        else:
            # Append the command to history
            #st.session_state.terminal_history.append(f"{command}")
            st.session_state.terminal_history[-1] += command

            # Execute the command and append output
            execute_command(command, st.session_state.terminal_history)
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")
            st.rerun()

    if st.button("Clear Terminal", use_container_width=True):
        st.session_state.terminal_history = ["$ "]
        st.rerun()

    st.subheader("Tool Shortcuts")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Scanning**")
        if st.button("Nmap Help", key="nmap", use_container_width=True):
            cmd = "nmap --help | head -n 20"
            if st.session_state.terminal_history:
                st.session_state.terminal_history[-1] += cmd
            else:
                st.session_state.terminal_history.append(f"[{datetime.now().strftime('%H:%M:%S')}] $ {cmd}")

            execute_command(cmd, st.session_state.terminal_history)
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")
            st.rerun()
    with col_b:
        st.markdown("**Web Tools**")
        if st.button("Dirsearch", key="dirsearch", use_container_width=True):
            cmd = "ls -la /tools | grep -i dirsearch"
            if st.session_state.terminal_history:
                st.session_state.terminal_history[-1] += cmd
            else:
                st.session_state.terminal_history.append(f"[{datetime.now().strftime('%H:%M:%S')}] $ {cmd}")

            execute_command(cmd, st.session_state.terminal_history)
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")
            st.rerun()
    with col_c:
        st.markdown("**System Info**")
        if st.button("System Status", key="system", use_container_width=True):
            cmd = "uname -a && df -h | head -n 2"
            if st.session_state.terminal_history:
                st.session_state.terminal_history[-1] += cmd
            else:
                st.session_state.terminal_history.append(f"[{datetime.now().strftime('%H:%M:%S')}] $ {cmd}")

            execute_command(cmd, st.session_state.terminal_history)
            st.session_state.terminal_history.append("")
            st.session_state.terminal_history.append("$ ")

            st.rerun()