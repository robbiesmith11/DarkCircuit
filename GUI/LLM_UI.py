import streamlit as st
import json
from datetime import datetime

def load_css():
    """
    Loads custom CSS from style.css if present in the same directory.
    If style.css doesn't exist, we skip custom styling.
    """
    try:
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Custom styling will be skipped.")

def create_copy_button(text, button_id):
    """
    Returns an HTML button that copies 'text' to the clipboard.
    Each button has a unique 'button_id' to avoid collisions in the DOM.
    """
    return f"""
        <button class="copy-button" onclick="copyText('{text}', this)" id="{button_id}">
            Copy to clipboard
        </button>
        <script>
            function copyText(text, button) {{
                navigator.clipboard.writeText(text);
                button.innerHTML = 'Copied!';
                setTimeout(() => button.innerHTML = 'Copy to clipboard', 2000);
            }}
        </script>
    """

def init_session_state():
    """
    Initializes the session state variables for messages and commands if not present.
    - 'messages': list of chat messages
    - 'commands': list of terminal commands
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'commands' not in st.session_state:
        st.session_state.commands = []

def add_message(role, content):
    """
    Appends a new chat message (role: 'user' or 'assistant') to st.session_state.messages.
    Includes a timestamp for display.
    """
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

def add_terminal_command(command):
    """
    Appends a new terminal command to st.session_state.commands.
    Does NOT add it to the chat messages, keeping chat and terminal separate.
    """
    formatted_command = f"$ {command}"
    st.session_state.commands.append(formatted_command)

def on_input_change():
    """
    Callback for the chat input box.
    Reads 'chat_input', adds it as a user message,
    and DOES NOT produce a duplicate or placeholder assistant message.
    """
    user_input = st.session_state.chat_input
    if user_input.strip():
        # Only add the user's message
        add_message("user", user_input)

        # If you do want an assistant response, uncomment or customize below:
        # response = "This is an assistant reply, not repeating user input."
        # add_message("assistant", response)

    # Clear chat input
    st.session_state.chat_input = ""

def on_terminal_input():
    """
    Callback for the terminal input box.
    Reads 'terminal_input', adds it to the terminal commands,
    and does NOT affect the chat container.
    """
    command = st.session_state.terminal_input
    if command.strip():
        add_terminal_command(command)

    # Clear terminal input
    st.session_state.terminal_input = ""

def main():
    st.set_page_config(page_title="Chat & Terminal Interface", layout="wide")
    load_css()
    init_session_state()

    # Create two columns: left for chat, right for terminal
    chat_col, terminal_col = st.columns(2)

    # ------------------
    # Chat Column (Left)
    # ------------------
    with chat_col:
        st.header("Chat Interface")

        # Display messages in chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            message_class = "user-message" if message["role"] == "user" else "bot-message"
            st.markdown(
                f"""
                <div class="chat-message {message_class}">
                    <div class="message-content">{message["content"]}</div>
                    <div class="message-timestamp">{message["timestamp"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Input box at the bottom for the chat
        st.text_input(
            "Type your message:",
            key="chat_input",
            on_change=on_input_change
        )

    # ----------------------
    # Terminal Column (Right)
    # ----------------------
    with terminal_col:
        st.header("Terminal")

        # Create terminal-like interface
        terminal_html = '<div class="terminal">'
        for i, command in enumerate(st.session_state.commands):
            terminal_html += f"""
            <div class="terminal-line">
                <code>{command}</code>
                {create_copy_button(command, f'copy-btn-{i}')}
            </div>
            """
        terminal_html += '</div>'

        st.markdown(terminal_html, unsafe_allow_html=True)

        # Add terminal input at the bottom
        st.text_input(
            "Enter command:",
            key="terminal_input",
            on_change=on_terminal_input
        )

        # Button to clear the terminal
        if st.button("Clear Terminal"):
            st.session_state.commands = []
            st.rerun()

if __name__ == "__main__":
    main()

