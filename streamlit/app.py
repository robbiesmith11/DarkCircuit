import subprocess

import streamlit as st
import requests
import paramiko
import time
import threading
import json

OLLAMA_API = "http://ollama:11434"
KALI_SSH_HOST = "kali"
KALI_SSH_PORT = 22
KALI_SSH_USER = "root"
KALI_SSH_PASS = "kali"

st.set_page_config(layout="wide")
st.title("AI Hacking Lab")

# ----------------------
# VPN Connection Section (Sidebar)
# ----------------------
st.sidebar.header("VPN Connection Setup")
vpn_file = st.sidebar.file_uploader("Upload your VPN (.ovpn) file", type=["ovpn"])


# Use session state to track VPN connection status
if "vpn_connected" not in st.session_state:
    st.session_state.vpn_connected = False

if not st.session_state.vpn_connected:
    if st.sidebar.button("Connect VPN"):
        if vpn_file:
            # Save VPN file to a known location in the container
            with open(VPN_FILE_PATH, "wb") as f:
                f.write(vpn_file.getbuffer())

            # Command to start OpenVPN in the Kali container
            # (Make sure the container name matches, e.g., "kali")
            connect_cmd = f"docker exec kali openvpn --config {VPN_FILE_PATH}"

            try:
                process = subprocess.Popen(connect_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = process.communicate(timeout=30)
                if error:
                    st.sidebar.error(f"VPN connection failed: {error.decode('utf-8')}")
                else:
                    st.sidebar.success("VPN connected successfully!")
                    st.session_state.vpn_connected = True

            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please upload a VPN file")
else:
    st.sidebar.success("VPN already connected.")

col1, col2 = st.columns(2)

### ✅ Dynamic Model Selection from Ollama
with col1:
    st.header("Select LLM Model")

    # Fetch available models from Ollama dynamically
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags")
        available_models = [model["name"] for model in response.json().get("models", [])]
    except Exception:
        available_models = ["No models available"]

    selected_model = st.selectbox("Choose a model:", available_models, key="selected_model")

    ### ✅ Chat UI with Real-time Updates
    st.header("Chat with LLM")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_window = st.empty()

    # Display chat history dynamically
    chat_text = "\n".join(st.session_state["chat_history"])
    chat_window.text_area("Chat Window", chat_text, height=300, disabled=True)

    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input and selected_model != "No models available":
            st.session_state["chat_history"].append(f"You: {user_input}")
            chat_text = "\n".join(st.session_state["chat_history"])
            chat_window.text_area("Chat Window", chat_text, height=300, disabled=True)

            # Send request to Ollama model with streaming enabled
            response = requests.post(
                f"{OLLAMA_API}/api/generate",
                json={"model": selected_model, "prompt": user_input, "stream": True},
                stream=True
            )

            ai_reply = ""
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as a JSON object
                        data = json.loads(line.decode("utf-8"))
                        ai_reply += data.get("response", "")
                    except json.JSONDecodeError:
                        continue  # Ignore malformed JSON parts

            st.session_state["chat_history"].append(f"AI: {ai_reply}")
            chat_text = "\n".join(st.session_state["chat_history"])
            chat_window.text_area("Chat Window", chat_text, height=300, disabled=True)

            st.rerun()

### ✅ Kali Terminal - Stream Output in Real-Time
with col2:
    st.header("Kali Terminal")

    # Terminal command input
    cmd = st.text_input("Command to run on Kali:", key="terminal_command")

    if st.button("Execute Command"):
        if cmd:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(KALI_SSH_HOST, port=KALI_SSH_PORT, username=KALI_SSH_USER, password=KALI_SSH_PASS)

                st.write(f"Executing: {cmd}")

                # Execute command without pty for cleaner output
                stdin, stdout, stderr = ssh.exec_command(cmd)

                # Wait for the command to complete
                exit_status = stdout.channel.recv_exit_status()

                # Get all output
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')

                # Initialize or update session state for terminal history
                if "terminal_history" not in st.session_state:
                    st.session_state.terminal_history = []

                # Add command to history
                st.session_state.terminal_history.append(f"$ {cmd}")

                # Add output to history (if any)
                if output:
                    st.session_state.terminal_history.append(output)

                # Add error to history (if any)
                if error:
                    st.session_state.terminal_history.append(f"ERROR: {error}")

                # Display full terminal history
                terminal_text = "\n".join(st.session_state.terminal_history)
                st.text_area("Terminal History", terminal_text, height=400)

                ssh.close()

            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {str(e)}")
        else:
            st.warning("Please enter a command to execute")

    # Option to clear terminal history
    if st.button("Clear Terminal History"):
        st.session_state.terminal_history = []
        st.rerun()

