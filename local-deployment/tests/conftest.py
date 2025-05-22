"""
Pytest configuration and fixtures for DarkCircuit tests.

This module provides shared fixtures and configuration for the test suite,
including mocks for external dependencies and test data setup.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Generator
import json

# Import components to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_utils import load_prompts, optimize_command
from streaming_handler import StreamingHandler
from Rag_tool import load_static_rag_context


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ssh_state():
    """Mock SSH connection state for testing."""
    return {
        "client": None,
        "channel": None,
        "connected": False,
        "error": None,
        "running": False
    }


@pytest.fixture
def mock_paramiko_client():
    """Mock paramiko SSH client for testing."""
    mock_client = Mock()
    mock_transport = Mock()
    mock_channel = Mock()
    
    # Configure mock behaviors
    mock_client.connect.return_value = None
    mock_client.get_transport.return_value = mock_transport
    mock_client.exec_command.return_value = (Mock(), Mock(), Mock())
    mock_transport.open_session.return_value = mock_channel
    mock_transport.is_active.return_value = True
    mock_transport.set_keepalive.return_value = None
    
    # Configure stdout mock for test command
    stdout_mock = Mock()
    stdout_mock.read.return_value = b"connected"
    mock_client.exec_command.return_value = (Mock(), stdout_mock, Mock())
    
    return mock_client


@pytest.fixture
def valid_ssh_credentials():
    """Valid SSH credentials for testing."""
    return {
        "host": "10.10.10.1",
        "port": 22,
        "username": "htb-student",
        "password": "test_password"
    }


@pytest.fixture
def invalid_ssh_credentials():
    """Invalid SSH credentials for testing."""
    return {
        "host": "10.10.10.1",
        "port": 22,
        "username": "invalid_user",
        "password": "wrong_password"
    }


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
        yield "test_api_key"


@pytest.fixture
def mock_langchain_components():
    """Mock LangChain components for agent testing."""
    with patch('darkcircuit_agent_modular.ChatOpenAI') as mock_llm, \
         patch('darkcircuit_agent_modular.DuckDuckGoSearchRun') as mock_search:
        
        # Configure mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance
        
        # Configure mock search
        mock_search_instance = Mock()
        mock_search.return_value = mock_search_instance
        
        yield {
            "llm": mock_llm_instance,
            "search": mock_search_instance
        }


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return {
        "reasonerPrompt": "You are a security expert. Analyze the situation step by step.",
        "responderPrompt": "Provide clear and helpful security guidance."
    }


@pytest.fixture
def sample_rag_documents():
    """Sample RAG documents for testing."""
    return [
        {
            "page_content": "This is a sample security document about SQL injection.",
            "metadata": {"source": "sql_injection_guide.pdf", "page": 1}
        },
        {
            "page_content": "Port scanning techniques using nmap.",
            "metadata": {"source": "nmap_guide.pdf", "page": 1}
        }
    ]


@pytest.fixture
def mock_faiss_vectorstore(sample_rag_documents):
    """Mock FAISS vector store for RAG testing."""
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents.return_value = [
        Mock(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in sample_rag_documents
    ]
    
    with patch('Rag_tool.FAISS') as mock_faiss:
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore
        yield mock_retriever


@pytest.fixture
def temp_docs_directory():
    """Create temporary docs directory with sample PDF."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample PDF file (empty file for testing)
        pdf_path = os.path.join(temp_dir, "sample_guide.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n")
        
        yield temp_dir


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for terminal testing."""
    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    mock_ws.receive = AsyncMock()
    mock_ws.close = AsyncMock()
    return mock_ws


@pytest.fixture
def sample_terminal_output():
    """Sample terminal output for command execution testing."""
    return {
        "simple_command": {
            "command": "pwd",
            "output": "/home/htb-student\n",
            "exit_code": 0
        },
        "command_with_error": {
            "command": "ls /nonexistent",
            "output": "ls: cannot access '/nonexistent': No such file or directory\n",
            "exit_code": 2
        },
        "long_running_command": {
            "command": "sleep 5 && echo done",
            "output": "done\n",
            "exit_code": 0
        }
    }


@pytest.fixture
def mock_streaming_handler():
    """Mock streaming handler for agent testing."""
    handler = Mock(spec=StreamingHandler)
    handler.queue = AsyncMock()
    handler.queue.put = AsyncMock()
    handler.stream = AsyncMock()
    handler.end = AsyncMock()
    return handler


@pytest.fixture
def sample_chat_request():
    """Sample chat request for API testing."""
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the current directory?"}
        ],
        "reasoner_prompt": None,
        "responder_prompt": None
    }


@pytest.fixture
def mock_command_optimization_cases():
    """Test cases for command optimization."""
    return [
        {
            "input": "nmap -sS target.com",
            "expected_output": "nmap -sS target.com --min-rate=1000",
            "expected_message": "Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution."
        },
        {
            "input": "nmap -A target.com",
            "expected_output": "nmap -A target.com --min-rate=1000",
            "expected_message": "Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution."
        },
        {
            "input": "nmap -p- target.com",
            "expected_output": "nmap -p- target.com --min-rate=1000",
            "expected_message": "Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution."
        },
        {
            "input": "ls -la",
            "expected_output": "ls -la",
            "expected_message": None
        },
        {
            "input": "nmap --min-rate=500 -sS target.com",
            "expected_output": "nmap --min-rate=500 -sS target.com",
            "expected_message": None
        }
    ]


@pytest.fixture
def sample_agent_responses():
    """Sample agent responses for testing."""
    return {
        "simple_query": {
            "query": "What is the current directory?",
            "expected_command": "pwd",
            "expected_response_contains": ["/home/", "current directory"]
        },
        "security_scan": {
            "query": "Scan target 10.10.10.5 for open ports",
            "expected_command": "nmap",
            "expected_response_contains": ["port", "scan", "10.10.10.5"]
        },
        "invalid_target": {
            "query": "Scan the target for vulnerabilities",
            "expected_command": None,
            "expected_response_contains": ["target IP", "configuration"]
        }
    }


# Test data constants
TEST_ANSI_SEQUENCES = [
    ("\x1b[31mRed text\x1b[0m", "Red text"),
    ("\x1b[1;32mBold green\x1b[0m", "Bold green"),
    ("\x1b[?25l\x1b[2J\x1b[H", ""),  # Hide cursor, clear screen, home
    ("Normal text", "Normal text"),
    ("", "")
]

TEST_CLI_PROMPTS = [
    ("└──╼ [★]$ ", True),
    ("smb: \\target\\> ", True),
    ("mysql> ", True),
    ("(gdb) ", True),
    ("[0x08048000]> ", True),
    ("Password: ", True),
    ("msf6 > ", True),
    ("normal output", False),
    ("still typing...", False)
]


@pytest.fixture
def ansi_test_data():
    """Test data for ANSI code stripping."""
    return TEST_ANSI_SEQUENCES


@pytest.fixture
def cli_prompt_test_data():
    """Test data for CLI prompt detection."""
    return TEST_CLI_PROMPTS


# Error simulation fixtures
@pytest.fixture
def ssh_connection_errors():
    """Different types of SSH connection errors for testing."""
    import paramiko
    return {
        "auth_failure": paramiko.ssh_exception.AuthenticationException("Authentication failed"),
        "connection_refused": paramiko.ssh_exception.NoValidConnectionsError({}),
        "timeout": TimeoutError("Connection timed out"),
        "general_error": Exception("General connection error")
    }


@pytest.fixture
def cleanup_files():
    """Cleanup fixture to remove test files after testing."""
    created_files = []
    
    def _create_file(path: str, content: str = ""):
        with open(path, "w") as f:
            f.write(content)
        created_files.append(path)
        return path
    
    yield _create_file
    
    # Cleanup
    for file_path in created_files:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass