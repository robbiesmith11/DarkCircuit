"""
DarkCircuit Agent - The main agent implementation.

This module provides the core agent functionality, leveraging LangChain
and LangGraph to create a sophisticated security-focused assistant
that can execute SSH commands and provide security guidance.
"""

import os
import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Import modularized components
from streaming_handler import StreamingHandler

from agent_utils import load_prompts
from Rag_tool import *
from utils import get_path

from dotenv import load_dotenv
load_dotenv(dotenv_path=get_path(".env"))

# Type definitions
MessagesState = dict

class Darkcircuit_Agent:
    """
    The main agent for DarkCircuit security assistant.

    This agent uses LangGraph to implement a reasoning process for security tasks,
    with the ability to execute commands on remote systems and provide analysis.
    """

    def __init__(self,
                 model_name="gpt-4o-mini",
                 reasoning_prompt=None,
                 response_prompt=None,
                 ssh_command_runner: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]] = None):
        """
        Initialize the agent.

        Args:
            model_name (str): The OpenAI model to use
            reasoning_prompt (str, optional): Custom prompt for the reasoning step
            response_prompt (str, optional): Custom prompt for the response step
            ssh_command_runner (callable, optional): Function to execute SSH commands directly
        """
        # Check for OpenAI API key in environment
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

        api_key = os.environ["OPENAI_API_KEY"]

        # Initialize the language model
        self.llm = ChatOpenAI(model=model_name, streaming=True, api_key=api_key)
        self.streaming_handler = None
        self.terminal_command_id = 0

        # Initialize agent tools
        self.ssh_command_runner = ssh_command_runner
        
        # Initialize terminal output queue for frontend communication
        if not self.ssh_command_runner:
            self.terminal_output_queue = asyncio.Queue()
        else:
            self.terminal_output_queue = None

        self.search = DuckDuckGoSearchRun()

        @tool
        def rag_retrieve(query: str) -> str:
            """Search for relevant documents included are writeups from hack the box challenges using RAG"""
            retriever = load_static_rag_context()
            docs = retriever.get_relevant_documents(query)
            content_parts = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                content_parts.append(f"[Source {i + 1}] {doc.page_content}")
            return "\n\n".join(content_parts)

        @tool
        def get_htb_techniques(service_or_situation: str = "") -> str:
            """Get immediate HTB exploitation techniques and commands for any service or situation"""
            
            techniques = {
                "ftp": [
                    "ftp <target_ip> → try anonymous:anonymous login",
                    "If anonymous access: download all files with 'mget *'",
                    "Check for writable directories: 'put test.txt' to test uploads",
                    "Look for config files, user files, .bash_history, id_rsa keys",
                    "Try common credentials: admin/admin, ftp/ftp, user/user"
                ],
                "ssh": [
                    "Try common credentials: admin/admin, root/root, user/user, guest/guest",
                    "Try usernames from web/ftp enumeration with passwords: password, 123456, admin",
                    "Check for SSH keys in web directories: wget http://<target_ip>/id_rsa",
                    "Look for .ssh directories in FTP or web access",
                    "Brute force if users found: hydra -l <user> -P /usr/share/wordlists/rockyou.txt ssh://<target_ip>"
                ],
                "web": [
                    "gobuster dir -u http://<target_ip> -w /usr/share/wordlists/dirb/common.txt -x php,html,txt,js",
                    "Check robots.txt, .htaccess, sitemap.xml, config.php, wp-config.php",
                    "Try directory traversal: curl http://<target_ip>/page?file=../../../etc/passwd",
                    "Test file upload: create PHP shell and upload if possible",
                    "Check source code for credentials, comments, hidden forms",
                    "Try common paths: /admin, /login, /config, /backup, /uploads",
                    "Look for databases: phpmyadmin, adminer, database backups"
                ],
                "smb": [
                    "smbclient -L <target_ip> → list shares (try null session)",
                    "smbclient //<target_ip>/share → connect to shares",
                    "enum4linux <target_ip> → enumerate users/groups/shares",
                    "Try guest access: smbclient //<target_ip>/share -U guest",
                    "Download everything: smbget -R smb://<target_ip>/share",
                    "Look for sensitive files: passwords.txt, config files, scripts"
                ],
                "flags": [
                    "find / -name 'user.txt' 2>/dev/null | head -5",
                    "find / -name 'root.txt' 2>/dev/null | head -5",
                    "find / -name '*flag*' 2>/dev/null | head -10",
                    "cat /root/root.txt 2>/dev/null || echo 'Need root access'",
                    "find /home -name 'user.txt' 2>/dev/null | xargs cat 2>/dev/null",
                    "find /var/www -name '*flag*' -o -name '*user*' -o -name '*root*' 2>/dev/null",
                    "grep -r 'HTB{' /var/www/ 2>/dev/null | head -5",
                    "find /opt -name '*flag*' 2>/dev/null"
                ],
                "privilege_escalation": [
                    "sudo -l → check sudo permissions",
                    "find / -perm -4000 2>/dev/null → SUID binaries",
                    "cat /etc/crontab → check cron jobs",
                    "ps aux → check running processes as root",
                    "netstat -tulpn → check listening services",
                    "cat /etc/passwd → enumerate users",
                    "ls -la /home → check user directories",
                    "find / -writable 2>/dev/null | grep -v proc | head -10",
                    "getcap -r / 2>/dev/null → check capabilities",
                    "cat /etc/crontab /etc/cron*/* 2>/dev/null | grep -v '#'"
                ],
                "reconnaissance": [
                    "nmap -p- --min-rate 10000 <target_ip>",
                    "nmap -sV -sC -p <ports> <target_ip>",
                    "nmap -sU --top-ports 100 <target_ip> (UDP scan)",
                    "whatweb <target_ip> (web technology detection)",
                    "nmap --script vuln -p <ports> <target_ip> (vulnerability scan)"
                ],
                "enumeration": [
                    "ls -la → check current directory permissions",
                    "whoami → current user",
                    "id → current user privileges",
                    "uname -a → system information",
                    "cat /etc/os-release → OS version",
                    "ps aux → running processes",
                    "netstat -tulpn → network connections",
                    "mount → mounted filesystems"
                ]
            }
            
            # Match service or situation
            service_lower = service_or_situation.lower()
            matched_techniques = []
            
            for category, commands in techniques.items():
                if category in service_lower or any(keyword in service_lower for keyword in category.split()):
                    matched_techniques.extend([f"🔥 {category.upper()}: {cmd}" for cmd in commands])
            
            # If no specific match, provide aggressive starting techniques
            if not matched_techniques:
                matched_techniques = [
                    "🚀 AGGRESSIVE START - EXECUTE ALL IMMEDIATELY:",
                    "• nmap -p- --min-rate 10000 <target_ip>",
                    "• nmap -sV -sC -p <discovered_ports> <target_ip>",
                    "• gobuster dir -u http://<target_ip> -w /usr/share/wordlists/dirb/common.txt -x php,txt,html",
                    "• ftp <target_ip> (try anonymous login)",
                    "• smbclient -L <target_ip>",
                    "• find / -name 'user.txt' 2>/dev/null",
                    "• find / -name 'root.txt' 2>/dev/null",
                    "• Try common web paths: /admin /login /config /backup"
                ]
            
            # Always add flag hunting section
            matched_techniques.extend([
                "",
                "🏁 IMMEDIATE FLAG HUNTING - RUN ALL:",
                "• find / -name 'user.txt' 2>/dev/null | xargs cat 2>/dev/null",
                "• find / -name 'root.txt' 2>/dev/null | xargs cat 2>/dev/null", 
                "• find / -name '*flag*' 2>/dev/null | head -10",
                "• find /var/www /home /opt -name '*flag*' -o -name '*user*' -o -name '*root*' 2>/dev/null",
                "• grep -r 'HTB{' /var/www/ /home/ 2>/dev/null | head -5",
                "",
                "🎯 PERSISTENCE STRATEGY:",
                "• Keep trying different attack vectors simultaneously",
                "• If one service blocked, immediately pivot to others",
                "• Always check for flags after each successful step",
                "• Don't stop until both user.txt and root.txt are found"
            ])
            
            return "\n".join(matched_techniques)

        @tool
        async def auto_enumerate(self, target_ip: str) -> str:
            """Automatically perform aggressive enumeration on target IP - runs multiple scans in parallel"""
            
            commands_to_run = [
                f"nmap -p- --min-rate 10000 {target_ip}",
                f"ftp {target_ip} <<< 'anonymous\nanonymous\nls\nquit'",
                f"smbclient -L {target_ip} -N",
                f"curl -s http://{target_ip}/robots.txt",
                f"curl -s http://{target_ip} | grep -i 'login\\|admin\\|config'",
                "find / -name 'user.txt' 2>/dev/null | head -3",
                "find / -name 'root.txt' 2>/dev/null | head -3"
            ]
            
            results = []
            for cmd in commands_to_run:
                try:
                    # Execute each enumeration command
                    result = await self.run_command.ainvoke(cmd)
                    results.append(f"[CMD: {cmd}]\n{result}\n")
                except Exception as e:
                    results.append(f"[CMD: {cmd}]\nError: {str(e)}\n")
            
            combined_results = "\n".join(results)
            return f"🔍 AGGRESSIVE AUTO-ENUMERATION RESULTS:\n\n{combined_results}\n\n🎯 NEXT: Analyze results and immediately exploit discovered services!"

        @tool
        async def hunt_flags(self) -> str:
            """Aggressively hunt for flags in common locations"""
            
            flag_commands = [
                "find / -name 'user.txt' 2>/dev/null | xargs cat 2>/dev/null",
                "find / -name 'root.txt' 2>/dev/null | xargs cat 2>/dev/null", 
                "find / -name '*flag*' 2>/dev/null | head -10",
                "find /var/www -name '*flag*' -o -name '*user*' -o -name '*root*' 2>/dev/null",
                "find /home -name '*flag*' -o -name 'user.txt' 2>/dev/null | xargs cat 2>/dev/null",
                "grep -r 'HTB{' /var/www/ /home/ /opt/ 2>/dev/null | head -5",
                "find /opt /srv /tmp -name '*flag*' -o -name '*user*' -o -name '*root*' 2>/dev/null"
            ]
            
            results = []
            flags_found = []
            
            for cmd in flag_commands:
                try:
                    result = await self.run_command.ainvoke(cmd)
                    results.append(f"[{cmd}]\n{result}\n")
                    
                    # Check if we found actual flags
                    if 'HTB{' in result or ('.txt' in result and len(result.strip()) > 0 and 'No such file' not in result):
                        flags_found.append(result.strip())
                        
                except Exception as e:
                    results.append(f"[{cmd}]\nError: {str(e)}\n")
            
            combined_results = "\n".join(results)
            
            if flags_found:
                flag_summary = "\n".join(flags_found)
                return f"🏆 FLAG HUNTING RESULTS - FLAGS FOUND! 🏆\n\n{flag_summary}\n\nFull Results:\n{combined_results}"
            else:
                return f"🔍 FLAG HUNTING RESULTS - No flags found yet, keep exploiting!\n\n{combined_results}\n\n💡 TIP: Try different privilege escalation techniques!"

        @tool
        async def run_command(command: str) -> str:
            """Execute a command on the remote SSH server (HTB Pwnbox environment)."""
            from agent_utils import optimize_command, wait_for_terminal_output

            try:
                # Generate a unique ID for this command execution
                self.terminal_command_id += 1
                command_id = self.terminal_command_id

                # Optimize the command if possible
                optimized_command, optimization_message = optimize_command(command)
                
                # Add context reminder and flag detection for all commands
                context_reminder = "\n[CONTEXT: You are on HTB Pwnbox executing commands against the target machine]"
                
                # Add flag hunting suggestion based on command type
                flag_hunting_hint = ""
                if any(cmd in optimized_command.lower() for cmd in ['ls', 'cat', 'find', 'grep', 'cd']):
                    flag_hunting_hint = "\n[HINT: After exploring, run: find / -name 'user.txt' -o -name 'root.txt' 2>/dev/null | xargs cat 2>/dev/null]"
                elif any(cmd in optimized_command.lower() for cmd in ['wget', 'curl', 'ftp', 'smbclient']):
                    flag_hunting_hint = "\n[HINT: After downloading files, check them for flags or credentials]"
                elif 'nmap' in optimized_command.lower():
                    flag_hunting_hint = "\n[HINT: After port scan, immediately test each service found]"

                # If we have a direct SSH command runner, use it
                if self.ssh_command_runner:
                    print(f"Executing command directly on SSH: {optimized_command} (ID: {command_id})")

                    # Still notify the frontend about the command for display purposes
                    if self.streaming_handler:
                        await self.streaming_handler.queue.put({
                            "type": "ui_terminal_command",
                            "command": optimized_command,
                            "command_id": command_id
                        })

                    # Execute the command directly using the provided function
                    result = await self.ssh_command_runner(optimized_command)

                    # Extract the output
                    if result and "success" in result:
                        output = result.get("output", "")

                        # Add optimization message if present
                        if optimization_message:
                            output = f"Note: {optimization_message}\n\n{output}"

                        # Add context reminder and flag hunting hints
                        output += context_reminder + flag_hunting_hint
                        
                        # Auto-detect potential flags in output
                        if 'HTB{' in output or 'user.txt' in output or 'root.txt' in output:
                            output += "\n\n🏆 POTENTIAL FLAG DETECTED IN OUTPUT! 🏆\n"

                        # Track command in learning system if available
                        if hasattr(self, 'learning_system') and self.learning_system:
                            try:
                                self.learning_system.track_command(optimized_command, output)
                                print(f"[Learning] Tracked command: {optimized_command[:50]}...")
                            except Exception as e:
                                print(f"[Learning] Error tracking command: {e}")

                        return output
                    else:
                        failure_msg = "Command execution failed or returned no results."
                        
                        # Track failed command in learning system if available
                        if hasattr(self, 'learning_system') and self.learning_system:
                            try:
                                self.learning_system.track_command(optimized_command, failure_msg, success=False)
                                print(f"[Learning] Tracked failed command: {optimized_command[:50]}...")
                            except Exception as e:
                                print(f"[Learning] Error tracking failed command: {e}")
                        
                        return failure_msg
                else:
                    # Fallback to the old method using the frontend
                    print(
                        f"No direct SSH runner available, using frontend for command: {optimized_command} (ID: {command_id})")

                    # Tell the frontend to execute the command
                    if self.streaming_handler:
                        await self.streaming_handler.queue.put({
                            "type": "ui_terminal_command",
                            "command": optimized_command,
                            "command_id": command_id
                        })

                    # Wait for output from the frontend
                    if self.terminal_output_queue:
                        output = await wait_for_terminal_output(self.terminal_output_queue, command_id)
                        
                        # Add optimization message if present
                        if optimization_message:
                            output = f"Note: {optimization_message}\n\n{output}"

                        # Add context reminder and flag hunting hints
                        output += context_reminder + flag_hunting_hint
                        
                        # Auto-detect potential flags in output
                        if 'HTB{' in output or 'user.txt' in output or 'root.txt' in output:
                            output += "\n\n🏆 POTENTIAL FLAG DETECTED IN OUTPUT! 🏆\n"
                        
                        # Track command in learning system if available
                        if hasattr(self, 'learning_system') and self.learning_system and output:
                            try:
                                self.learning_system.track_command(optimized_command, output)
                                print(f"[Learning] Tracked frontend command: {optimized_command[:50]}...")
                            except Exception as e:
                                print(f"[Learning] Error tracking frontend command: {e}")
                        
                        return output
                    else:
                        return "Terminal output queue not initialized."

            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.run_command = run_command
        self.rag_retrieve = rag_retrieve
        self.get_htb_techniques = get_htb_techniques
        self.tools = [self.search, self.run_command, self.rag_retrieve, self.get_htb_techniques]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Load default prompts from file or use hardcoded defaults
        DEFAULT_REASONING_PROMPT, DEFAULT_RESPONSE_PROMPT = load_prompts()

        # Create system messages
        self.reasoning_prompt = SystemMessage(content=reasoning_prompt or DEFAULT_REASONING_PROMPT)
        self.response_prompt = SystemMessage(content=response_prompt or DEFAULT_RESPONSE_PROMPT)
        
        # Preload RAG system during agent initialization for faster response times
        print("🔄 Initializing RAG system for instant responses...")
        from Rag_tool import preload_rag_system
        preload_rag_system()

    def _build_agent_graph(self):
        """Set up the LangGraph for agent reasoning."""
        builder = StateGraph(MessagesState)

        # Add nodes
        builder.add_node("reasoner", self.reasoner)
        builder.add_node("tools", self.tools_node)
        builder.add_node("responder", self.responder)

        # Add edges
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", self.route_from_reasoner)
        builder.add_edge("tools", "reasoner")
        builder.add_edge("responder", END)

        # Compile the graph
        self.react_graph = builder.compile()

    async def receive_terminal_output(self, output_data):
        """
        Receive terminal output from the frontend.

        Args:
            output_data (dict): Output data from the terminal
        """
        # Find the run_command tool and pass the output to its queue
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == 'run_command':
                if hasattr(tool, 'terminal_output_queue'):
                    await tool.terminal_output_queue.put(output_data)
                break

    async def reasoner(self, state: MessagesState):
        """
        Agent reasoning node - determines what actions to take.

        Args:
            state (dict): Current state of the conversation

        Returns:
            dict: Updated state with reasoning results
        """
        print("[Agent] Entering reasoner node with message count:", len(state["messages"]))

        # Use debug output target for reasoner
        self.streaming_handler.output_target = "debug"

        try:
            # When preparing messages for the reasoner, include the original user query
            # and system messages, but convert tool messages to system messages
            filtered_messages = []
            has_htb_request = False
            
            for msg in state["messages"]:
                if isinstance(msg, (HumanMessage, SystemMessage)):
                    filtered_messages.append(msg)
                    # Check if this is an HTB challenge request
                    if isinstance(msg, HumanMessage) and any(keyword in msg.content.lower() for keyword in ['htb', 'hack the box', 'target', 'machine', 'challenge', 'dancing', 'meow', 'fawn', 'explosion']):
                        has_htb_request = True
                elif isinstance(msg, ToolMessage):
                    # Convert tool messages to system messages to preserve their content
                    tool_name = getattr(msg, "name", "tool")
                    tool_content = getattr(msg, "content", "")
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {tool_content}")
                    filtered_messages.append(system_msg)
                elif getattr(msg, "type", "") in ["human", "system"]:
                    filtered_messages.append(msg)
                    if getattr(msg, "type", "") == "human" and any(keyword in msg.content.lower() for keyword in ['htb', 'hack the box', 'target', 'machine', 'challenge', 'dancing', 'meow', 'fawn', 'explosion']):
                        has_htb_request = True
                elif getattr(msg, "type", "") == "tool":
                    tool_name = getattr(msg, "name", "tool")
                    tool_content = getattr(msg, "content", "")
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {tool_content}")
                    filtered_messages.append(system_msg)
            
            # If this looks like an HTB challenge, inject aggressive action guidance
            if has_htb_request:
                if len(state["messages"]) < 5:
                    # Initial aggressive push
                    action_boost = SystemMessage(content="🚨 IMMEDIATE ACTION REQUIRED: This is an HTB challenge. Start MULTIPLE reconnaissance approaches NOW - don't overthink, execute in parallel: 1) nmap -p- --min-rate 10000 <target_ip> 2) If web services found, immediately run gobuster 3) Try FTP anonymous login 4) Check SMB shares 5) Hunt for flags constantly")
                    filtered_messages.append(action_boost)
                elif len(state["messages"]) < 15:
                    # Mid-challenge persistence boost
                    persistence_boost = SystemMessage(content="🔄 PERSISTENCE MODE: You should be aggressively trying multiple attack vectors. If stuck on one service, immediately pivot to others. ALWAYS check for flags after any successful access. Keep the pressure up!")
                    filtered_messages.append(persistence_boost)
                
                # Always add flag hunting reminder
                flag_reminder = SystemMessage(content="FLAG HUNTING: After EVERY successful command, immediately run: find / -name 'user.txt' 2>/dev/null | xargs cat 2>/dev/null && find / -name 'root.txt' 2>/dev/null | xargs cat 2>/dev/null")
                filtered_messages.append(flag_reminder)

            # If no messages after filtering, check for the original human message
            if not filtered_messages:
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                        filtered_messages.append(msg)
                        break

            # Add the system prompt
            messages_to_send = [self.reasoning_prompt] + filtered_messages

            # Log what we're sending
            print(f"[Agent] Reasoner sending {len(messages_to_send)} messages")
            for i, msg in enumerate(messages_to_send):
                content = getattr(msg, "content", "")
                print(f"[Agent] Message {i}: {type(msg).__name__} - {content[:50]}...")

            # Only invoke once with a timeout
            result = await asyncio.wait_for(
                self.llm_with_tools.ainvoke(
                    messages_to_send,
                    config={"callbacks": [self.streaming_handler]}
                ),
                timeout=60  # Add a timeout to prevent hanging
            )

            # Extract and log the content
            result_text = getattr(result, "content", "").strip().lower()
            print(f"[Agent] Reasoner result preview: {result_text[:100]}...")

            # Original simple logic - determine if we're done based on the magic phrase
            done = "[ready to answer]" in result_text
            print(f"[Agent] Done status: {done}")
            
            # Update the state with filtered messages and the result
            new_messages = filtered_messages + [result]

            # Flush the thinking buffer to send consolidated thinking content
            await self.streaming_handler._flush_thinking_buffer(done)

            return {**state, "messages": new_messages, "done": done}

        except Exception as e:
            print(f"[Agent] Error in reasoner: {str(e)}")
            # Handle error and still return something valid

            # Flush any buffered thinking content before sending the error
            await self.streaming_handler._flush_thinking_buffer()

            await self.streaming_handler.queue.put({
                "type": "thinking",
                "value": f"Error in reasoning: {str(e)}"
            })

            # Return original messages to avoid corrupting state
            return {**state, "messages": state["messages"], "done": True}  # Force done to exit on error

    async def tools_node(self, state: MessagesState):
        """
        Execute tool calls from the reasoner.

        Args:
            state (dict): Current state with tool calls

        Returns:
            dict: Updated state with tool results
        """
        print("[Agent] Entering tools node")

        # Get the last message which should contain the tool calls
        last_message = state["messages"][-1] if state["messages"] else None

        # Extract tool calls from the message
        tool_calls = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls

        # Process each tool call sequentially instead of using ToolNode
        preserved_messages = list(state["messages"])

        # Process tool calls one at a time
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown_tool')
            tool_args = tool_call.get('args', {})

            print(f"[Agent] Processing tool call: {tool_name} with args: {tool_args}")

            # Find the matching tool
            matching_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    matching_tool = tool
                    break

            if matching_tool:
                try:
                    # Execute the tool with appropriate arguments
                    if tool_name == "run_command":
                        # Extract command string from arguments
                        command = ""
                        if isinstance(tool_args, dict):
                            if 'command' in tool_args:
                                command = tool_args['command']
                            # Some models may use 'input' instead of 'command'
                            elif 'input' in tool_args:
                                command = tool_args['input']
                            # Handle case where the tool_args might be incorrectly formatted
                            elif len(tool_args) == 1 and 'self' in tool_args:
                                # Try to extract command from the 'self' field if it's a string
                                if isinstance(tool_args['self'], str):
                                    command = tool_args['self']
                                else:
                                    raise ValueError("Invalid command format in tool arguments")
                        elif isinstance(tool_args, str):
                            command = tool_args
                        else:
                            command = str(tool_args)

                        print(f"[Agent] Executing command: {command}")

                        # Use ainvoke with the extracted command
                        result = await matching_tool.ainvoke(command)
                    else:
                        # For other tools, execute with standard arguments
                        # Make sure we handle cases where the tool might be expecting a specific argument name
                        if isinstance(tool_args, dict):
                            # If this is a properly formatted dict with expected argument names, use it directly
                            if not (len(tool_args) == 1 and 'self' in tool_args):
                                result = await matching_tool.ainvoke(tool_args)
                            # If it's a dict with just 'self', extract that value
                            elif isinstance(tool_args['self'], str):
                                result = await matching_tool.ainvoke(tool_args['self'])
                            else:
                                result = await matching_tool.ainvoke({})
                        elif isinstance(tool_args, str):
                            # If it's just a string, pass it directly
                            result = await matching_tool.ainvoke(tool_args)
                        else:
                            # Otherwise, convert to string
                            result = await matching_tool.ainvoke(str(tool_args))

                    # Convert result to a tool message
                    tool_message = ToolMessage(
                        content=result,
                        name=tool_name,
                        tool_call_id=tool_call.get('id', '')
                    )

                    # Also create a system message version for the agent to understand
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {result}")

                    # Add both messages to preserved messages
                    preserved_messages.append(system_msg)
                    preserved_messages.append(tool_message)

                    print(f"[Agent] Tool {tool_name} completed with result: {str(result)[:100]}...")

                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"[Agent] {error_msg}")
                    import traceback
                    traceback.print_exc()

                    # Create error messages
                    error_tool_message = ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=tool_call.get('id', '')
                    )
                    error_system_msg = SystemMessage(content=f"Tool error from {tool_name}: {error_msg}")

                    # Add error messages
                    preserved_messages.append(error_system_msg)
                    preserved_messages.append(error_tool_message)
            else:
                error_msg = f"Tool {tool_name} not found"
                print(f"[Agent] {error_msg}")

                # Create not found messages
                not_found_tool_message = ToolMessage(
                    content=error_msg,
                    name=tool_name,
                    tool_call_id=tool_call.get('id', '')
                )
                not_found_system_msg = SystemMessage(content=f"Tool error: {error_msg}")

                # Add not found messages
                preserved_messages.append(not_found_system_msg)
                preserved_messages.append(not_found_tool_message)

        # Update the state with preserved messages
        return {**state, "messages": preserved_messages}

    async def responder(self, state: MessagesState):
        """
        Generate the final response to the user.

        Args:
            state (dict): Current state with reasoning results

        Returns:
            dict: Updated state with the final response
        """
        print("[Agent] Entering responder node")

        # Flush any remaining thinking content from the reasoner
        await self.streaming_handler._flush_thinking_buffer()

        # Use chat output target for responder - this will show in chat history
        self.streaming_handler.output_target = "chat"

        try:
            # Filter messages for the responder
            # Include the original user query and system messages (including our converted tool results)
            filtered_messages = []
            human_message = None
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    # Keep track of the original human message
                    human_message = msg
                    filtered_messages.append(msg)
                elif isinstance(msg, SystemMessage):
                    filtered_messages.append(msg)
                elif getattr(msg, "type", "") == "human":
                    human_message = msg
                    filtered_messages.append(msg)
                elif getattr(msg, "type", "") == "system":
                    filtered_messages.append(msg)

            # Ensure we have the human message
            if not human_message and len(state["messages"]) > 0:
                # Try to find the first human message in the original state
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                        human_message = msg
                        if human_message not in filtered_messages:
                            filtered_messages.append(human_message)
                        break

            # Add the response prompt
            messages_to_send = [self.response_prompt] + filtered_messages

            # Log what we're sending to the responder
            print(f"[Agent] Responder sending {len(messages_to_send)} messages")
            for i, msg in enumerate(messages_to_send):
                content = getattr(msg, "content", "")
                print(f"[Agent] Message {i}: {type(msg).__name__} - {content[:50]}...")

            # Generate the final response
            result = await self.llm.ainvoke(
                messages_to_send,
                config={"callbacks": [self.streaming_handler]}
            )

            # No need to manually send events - callback handles it
            new_messages = state["messages"] + [result]
            return {**state, "messages": new_messages}

        except Exception as e:
            print(f"[Agent] Error in responder: {str(e)}")
            # Since this is the last node, output error to chat
            await self.streaming_handler.queue.put({
                "type": "token",
                "value": f"I apologize, but I encountered an error while processing your request. Please try again."
            })
            return state

    def route_from_reasoner(self, state):
        """
        Determine the next step after reasoning.

        Args:
            state (dict): Current state

        Returns:
            str: Next node to execute ("tools" or "responder")
        """
        print(f"[Agent] Routing from reasoner. Done: {state.get('done')}")

        # If there was an error or we're explicitly done, go to responder
        if state.get('done', False):
            return "responder"

        # Check if the last message contains tool calls
        last_message = state["messages"][-1] if state["messages"] else None
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            return "tools"  # Process tool calls
        else:
            # No tool calls but not done yet - go to responder as fallback
            return "responder"

    async def run_agent_streaming(self, prompt: str):
        """
        Run the agent with streaming output.

        Args:
            prompt (str): The user's prompt/query

        Yields:
            dict: Event objects for streaming (tokens, thinking, tool calls, etc.)
        """
        input_messages = [HumanMessage(content=prompt)]

        # Initialize streaming handler
        self.streaming_handler = StreamingHandler(output_target="debug")

        # Build the agent graph
        self._build_agent_graph()

        async def run_graph():
            print("[Agent] Starting graph execution")
            await self.react_graph.ainvoke(
                {"messages": input_messages},
                config={"callbacks": [self.streaming_handler]}
            )
            await self.streaming_handler.end()
            print("[Agent] Graph execution complete")

        graph_task = asyncio.create_task(run_graph())

        try:
            async for event in self.streaming_handler.stream():
                yield event
        finally:
            await graph_task