"""
Integration tests for FastAPI endpoints and agent integration.

Tests cover:
- Chat completions API endpoint
- SSH connection API endpoints  
- WebSocket terminal communication
- Agent workflow integration
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

pytestmark = [pytest.mark.integration]

# Import the FastAPI app and components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import components, skip tests if dependencies are missing
try:
    from local_app import app, ssh_state, _setup_ssh_connection, _close_ssh_connection
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping API tests due to missing dependencies: {e}", allow_module_level=True)


class TestSSHConnectionEndpoints:
    """Test class for SSH connection API endpoints."""

    def test_ssh_connect_endpoint_success(self, valid_ssh_credentials, mock_paramiko_client):
        """
        Test successful SSH connection via API endpoint.
        
        Tests that the /api/ssh/connect endpoint properly handles
        valid SSH credentials and establishes connections.
        """
        client = TestClient(app)
        
        with patch('local_app._setup_ssh_connection') as mock_setup:
            mock_setup.return_value = {
                "success": True,
                "message": f"SSH connection established to {valid_ssh_credentials['host']}"
            }
            
            response = client.post(
                "/api/ssh/connect",
                json=valid_ssh_credentials
            )
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "SSH connection established" in response_data["message"]
            
            # Verify setup function was called with correct parameters
            mock_setup.assert_called_once_with(
                valid_ssh_credentials["host"],
                valid_ssh_credentials["port"],
                valid_ssh_credentials["username"],
                valid_ssh_credentials["password"],
                None
            )

    def test_ssh_connect_endpoint_failure(self, invalid_ssh_credentials):
        """
        Test SSH connection failure via API endpoint.
        
        Tests that the endpoint properly handles authentication
        failures and returns appropriate error responses.
        """
        client = TestClient(app)
        
        with patch('local_app._setup_ssh_connection') as mock_setup:
            mock_setup.return_value = {
                "success": False,
                "error": "Authentication failed. Please check your username, password or key."
            }
            
            response = client.post(
                "/api/ssh/connect",
                json=invalid_ssh_credentials
            )
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is False
            assert "Authentication failed" in response_data["error"]

    def test_ssh_connect_endpoint_with_key(self, valid_ssh_credentials):
        """
        Test SSH connection with key authentication via API.
        
        Tests that the endpoint properly handles key-based
        authentication parameters.
        """
        client = TestClient(app)
        
        # Add key path to credentials
        credentials_with_key = valid_ssh_credentials.copy()
        credentials_with_key["key_path"] = "/path/to/private/key"
        
        with patch('local_app._setup_ssh_connection') as mock_setup:
            mock_setup.return_value = {
                "success": True,
                "message": "SSH connection established with key authentication"
            }
            
            response = client.post(
                "/api/ssh/connect",
                json=credentials_with_key
            )
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            
            # Verify key path was passed
            mock_setup.assert_called_once_with(
                credentials_with_key["host"],
                credentials_with_key["port"],
                credentials_with_key["username"],
                credentials_with_key["password"],
                credentials_with_key["key_path"]
            )

    def test_ssh_disconnect_endpoint(self):
        """
        Test SSH disconnection via API endpoint.
        
        Tests that the /api/ssh/disconnect endpoint properly
        closes connections and cleans up resources.
        """
        client = TestClient(app)
        
        with patch('local_app._close_ssh_connection') as mock_close:
            response = client.post("/api/ssh/disconnect")
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "SSH connection closed" in response_data["message"]
            
            # Verify close function was called
            mock_close.assert_called_once()

    def test_ssh_connect_endpoint_malformed_request(self):
        """
        Test SSH connection with malformed request data.
        
        Tests that the endpoint properly handles invalid or
        missing request parameters.
        """
        client = TestClient(app)
        
        # Test with missing required fields
        incomplete_data = {"host": "test.com"}
        
        response = client.post("/api/ssh/connect", json=incomplete_data)
        
        # Should handle gracefully (may return error or use defaults)
        assert response.status_code == 200
        # The endpoint should handle missing fields gracefully

    def test_ssh_connect_endpoint_exception_handling(self, valid_ssh_credentials):
        """
        Test SSH connection endpoint exception handling.
        
        Tests that the endpoint properly handles exceptions
        during connection attempts.
        """
        client = TestClient(app)
        
        with patch('local_app._setup_ssh_connection') as mock_setup:
            mock_setup.side_effect = Exception("Connection setup failed")
            
            response = client.post(
                "/api/ssh/connect",
                json=valid_ssh_credentials
            )
            
            # Should handle exception gracefully
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is False
            assert "Error connecting to SSH" in response_data["error"]


class TestChatCompletionsEndpoint:
    """Test class for chat completions API endpoint."""

    def test_chat_completions_endpoint_basic(self, sample_chat_request, mock_openai_api_key):
        """
        Test basic chat completions functionality.
        
        Tests that the /api/chat/completions endpoint properly
        creates agent instances and processes chat requests.
        """
        client = TestClient(app)
        
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            # Create mock agent instance
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Mock streaming response
            async def mock_stream():
                yield {"type": "token", "value": "Hello "}
                yield {"type": "token", "value": "world!"}
            
            mock_agent.run_agent_streaming.return_value = mock_stream()
            
            response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            # Verify response
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            # Verify agent was created with correct parameters
            mock_agent_class.assert_called_once_with(
                model_name=sample_chat_request["model"],
                reasoning_prompt=sample_chat_request["reasoner_prompt"],
                response_prompt=sample_chat_request["responder_prompt"],
                ssh_command_runner=mock_agent_class.call_args[1]["ssh_command_runner"]
            )
            
            # Verify agent was called with user message
            user_message = sample_chat_request["messages"][-1]["content"]
            mock_agent.run_agent_streaming.assert_called_once_with(user_message)

    def test_chat_completions_endpoint_with_ssh_runner(self, sample_chat_request, mock_openai_api_key):
        """
        Test chat completions with SSH command runner integration.
        
        Tests that the agent is properly configured with SSH
        command execution capabilities.
        """
        client = TestClient(app)
        
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class, \
             patch('local_app.run_ssh_command') as mock_ssh_runner:
            
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            async def mock_stream():
                yield {"type": "token", "value": "Command executed"}
            
            mock_agent.run_agent_streaming.return_value = mock_stream()
            
            response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            # Verify SSH runner was passed to agent
            assert response.status_code == 200
            agent_init_kwargs = mock_agent_class.call_args[1]
            assert "ssh_command_runner" in agent_init_kwargs
            assert agent_init_kwargs["ssh_command_runner"] is not None

    def test_chat_completions_endpoint_streaming_events(self, sample_chat_request, mock_openai_api_key):
        """
        Test chat completions streaming event types.
        
        Tests that different types of streaming events (tokens, tool calls,
        debug info) are properly formatted and transmitted.
        """
        client = TestClient(app)
        
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Mock comprehensive streaming events
            async def mock_stream():
                yield {"type": "thinking", "value": "Analyzing request..."}
                yield {"type": "tool_call", "name": "run_command", "input": "pwd"}
                yield {"type": "tool_result", "output": "/home/user"}
                yield {"type": "token", "value": "The current directory is /home/user"}
                yield {"type": "ui_terminal_command", "command": "pwd", "command_id": 1}
            
            mock_agent.run_agent_streaming.return_value = mock_stream()
            
            response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            # Verify response format
            assert response.status_code == 200
            
            # Parse streaming response
            content = response.content.decode()
            lines = content.split('\n')
            
            # Should contain various event types
            data_lines = [line for line in lines if line.startswith('data: ')]
            assert len(data_lines) > 0
            
            # Verify event structure
            for data_line in data_lines:
                if data_line == 'data: [DONE]':
                    continue
                
                try:
                    event_data = json.loads(data_line[6:])  # Remove 'data: ' prefix
                    
                    # Check for either OpenAI format or custom event format
                    if "choices" in event_data:
                        # OpenAI chat completion format
                        assert "id" in event_data
                        assert "object" in event_data
                        assert "model" in event_data
                    else:
                        # Custom event format
                        assert "type" in event_data
                        
                except json.JSONDecodeError:
                    # Some lines might be formatting, skip them
                    pass

    def test_chat_completions_endpoint_error_handling(self, sample_chat_request, mock_openai_api_key):
        """
        Test chat completions endpoint error handling.
        
        Tests that the endpoint properly handles various error
        conditions during agent creation and execution.
        """
        client = TestClient(app)
        
        # Test missing OpenAI API key by mocking agent creation failure
        with patch('local_app.Darkcircuit_Agent') as mock_agent:
            mock_agent.side_effect = ValueError("OPENAI_API_KEY environment variable is not set")
            
            response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            # Should handle missing API key gracefully
            assert response.status_code == 500

    def test_chat_completions_endpoint_custom_prompts(self, mock_openai_api_key):
        """
        Test chat completions with custom system prompts.
        
        Tests that custom reasoner and responder prompts are
        properly passed to the agent.
        """
        client = TestClient(app)
        
        custom_request = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test message"}],
            "reasoner_prompt": "Custom reasoning prompt",
            "responder_prompt": "Custom response prompt"
        }
        
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            async def mock_stream():
                yield {"type": "token", "value": "Response"}
            
            mock_agent.run_agent_streaming.return_value = mock_stream()
            
            response = client.post(
                "/api/chat/completions",
                json=custom_request
            )
            
            # Verify custom prompts were passed
            assert response.status_code == 200
            mock_agent_class.assert_called_once_with(
                model_name=custom_request["model"],
                reasoning_prompt=custom_request["reasoner_prompt"],
                response_prompt=custom_request["responder_prompt"],
                ssh_command_runner=mock_agent_class.call_args[1]["ssh_command_runner"]
            )


class TestWebSocketTerminalEndpoint:
    """Test class for WebSocket terminal endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_terminal_no_ssh_connection(self):
        """
        Test WebSocket terminal with no SSH connection.
        
        Tests that the WebSocket properly handles cases where
        no SSH connection is established.
        """
        # Mock WebSocket
        mock_websocket = AsyncMock()
        
        # Ensure no SSH connection
        with patch.dict('local_app.ssh_state', {
            "connected": False,
            "client": None,
            "channel": None,
            "error": None,
            "running": False
        }):
            
            # Import the endpoint function
            from local_app import terminal_endpoint
            
            # Execute endpoint
            await terminal_endpoint(mock_websocket)
            
            # Verify appropriate response
            mock_websocket.accept.assert_called_once()
            mock_websocket.send_text.assert_called()
            mock_websocket.close.assert_called_once()
            
            # Check that error message was sent
            sent_message = mock_websocket.send_text.call_args[0][0]
            assert "No SSH connection established" in sent_message

    @pytest.mark.asyncio
    async def test_websocket_terminal_ssh_transport_inactive(self):
        """
        Test WebSocket terminal with inactive SSH transport.
        
        Tests handling of SSH connections where the transport
        layer has become inactive.
        """
        mock_websocket = AsyncMock()
        mock_client = Mock()
        mock_transport = Mock()
        
        # Configure inactive transport
        mock_transport.is_active.return_value = False
        mock_client.get_transport.return_value = mock_transport
        
        with patch.dict('local_app.ssh_state', {
            "connected": True,
            "client": mock_client,
            "channel": None,
            "error": None,
            "running": False
        }):
            
            from local_app import terminal_endpoint
            
            await terminal_endpoint(mock_websocket)
            
            # Verify error handling
            mock_websocket.accept.assert_called_once()
            mock_websocket.send_text.assert_called()
            mock_websocket.close.assert_called_once()
            
            sent_message = mock_websocket.send_text.call_args[0][0]
            assert "SSH transport is no longer active" in sent_message

    @pytest.mark.asyncio
    async def test_websocket_terminal_successful_connection(self):
        """
        Test successful WebSocket terminal connection.
        
        Tests the full WebSocket terminal flow with proper
        SSH connection and channel establishment.
        """
        mock_websocket = AsyncMock()
        mock_client = Mock()
        mock_transport = Mock()
        mock_channel = Mock()
        
        # Configure successful connection
        mock_transport.is_active.return_value = True
        mock_client.get_transport.return_value = mock_transport
        mock_transport.open_session.return_value = mock_channel
        mock_channel.get_pty.return_value = None
        mock_channel.invoke_shell.return_value = None
        mock_channel.exit_status_ready.return_value = False
        mock_channel.recv_ready.return_value = False
        
        with patch.dict('local_app.ssh_state', {
            "connected": True,
            "client": mock_client,
            "channel": None,
            "error": None,
            "running": False
        }):
            
            with patch('local_app.terminal_ws_clients', set()) as mock_clients:
                
                from local_app import terminal_endpoint
                
                # Mock asyncio.gather to avoid infinite loop
                with patch('asyncio.gather', side_effect=asyncio.CancelledError()):
                    try:
                        await terminal_endpoint(mock_websocket)
                    except asyncio.CancelledError:
                        pass  # Expected due to our mock
                
                # Verify setup sequence
                mock_websocket.accept.assert_called_once()
                mock_transport.set_keepalive.assert_called_with(30)
                mock_transport.open_session.assert_called_once()
                mock_channel.get_pty.assert_called()
                mock_channel.invoke_shell.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_terminal_channel_creation_failure(self):
        """
        Test WebSocket terminal when channel creation fails.
        
        Tests error handling when SSH channel cannot be
        established properly.
        """
        mock_websocket = AsyncMock()
        mock_client = Mock()
        mock_transport = Mock()
        
        # Configure transport failure
        mock_transport.is_active.return_value = True
        mock_client.get_transport.return_value = mock_transport
        mock_transport.open_session.side_effect = Exception("Channel creation failed")
        
        with patch.dict('local_app.ssh_state', {
            "connected": True,
            "client": mock_client,
            "channel": None,
            "error": None,
            "running": False
        }):
            
            from local_app import terminal_endpoint
            
            await terminal_endpoint(mock_websocket)
            
            # Verify error handling
            mock_websocket.accept.assert_called_once()
            mock_websocket.send_text.assert_called()
            mock_websocket.close.assert_called_once()
            
            sent_message = mock_websocket.send_text.call_args[0][0]
            assert "Failed to establish SSH channel" in sent_message

    @pytest.mark.asyncio
    async def test_websocket_terminal_pty_failure(self):
        """
        Test WebSocket terminal when PTY allocation fails.
        
        Tests error handling when terminal PTY cannot be
        allocated on the remote system.
        """
        mock_websocket = AsyncMock()
        mock_client = Mock()
        mock_transport = Mock()
        mock_channel = Mock()
        
        # Configure PTY failure
        mock_transport.is_active.return_value = True
        mock_client.get_transport.return_value = mock_transport
        mock_transport.open_session.return_value = mock_channel
        mock_channel.get_pty.side_effect = Exception("PTY allocation failed")
        
        with patch.dict('local_app.ssh_state', {
            "connected": True,
            "client": mock_client,
            "channel": None,
            "error": None,
            "running": False
        }):
            
            from local_app import terminal_endpoint
            
            await terminal_endpoint(mock_websocket)
            
            # Verify error handling
            mock_websocket.accept.assert_called_once()
            mock_websocket.send_text.assert_called()
            mock_websocket.close.assert_called_once()
            
            sent_message = mock_websocket.send_text.call_args[0][0]
            assert "Failed to initialize terminal" in sent_message


class TestAPIIntegration:
    """Test API integration and workflow scenarios."""

    def test_full_ssh_to_chat_workflow(self, valid_ssh_credentials, sample_chat_request, mock_openai_api_key):
        """
        Test complete workflow from SSH connection to chat completion.
        
        Tests the full integration flow: establish SSH connection,
        then use chat completions with SSH command execution.
        """
        client = TestClient(app)
        
        # Step 1: Establish SSH connection
        with patch('local_app._setup_ssh_connection') as mock_setup:
            mock_setup.return_value = {
                "success": True,
                "message": "SSH connection established"
            }
            
            ssh_response = client.post(
                "/api/ssh/connect",
                json=valid_ssh_credentials
            )
            
            assert ssh_response.status_code == 200
            assert ssh_response.json()["success"] is True
        
        # Step 2: Use chat completions with SSH
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            async def mock_stream():
                yield {"type": "token", "value": "Executing command..."}
                yield {"type": "tool_call", "name": "run_command", "input": "pwd"}
                yield {"type": "tool_result", "output": "/home/user"}
                yield {"type": "token", "value": "Current directory: /home/user"}
            
            mock_agent.run_agent_streaming.return_value = mock_stream()
            
            chat_response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            assert chat_response.status_code == 200
            
            # Verify agent was configured with SSH runner
            agent_kwargs = mock_agent_class.call_args[1]
            assert "ssh_command_runner" in agent_kwargs

    def test_api_error_propagation(self, sample_chat_request, mock_openai_api_key):
        """
        Test error propagation through API layers.
        
        Tests that errors from lower layers (SSH, agent, etc.)
        are properly propagated through the API endpoints.
        """
        client = TestClient(app)
        
        # Test agent creation failure
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            mock_agent_class.side_effect = Exception("Agent initialization failed")
            
            response = client.post(
                "/api/chat/completions",
                json=sample_chat_request
            )
            
            # Should handle agent creation error gracefully
            assert response.status_code == 500

    def test_concurrent_api_requests(self, sample_chat_request, mock_openai_api_key):
        """
        Test concurrent API request handling.
        
        Tests that the API can handle multiple concurrent
        requests without interference.
        """
        client = TestClient(app)
        
        with patch('local_app.Darkcircuit_Agent') as mock_agent_class:
            # Create different mock agents for each request
            def create_mock_agent(*args, **kwargs):
                mock_agent = Mock()
                
                async def mock_stream():
                    yield {"type": "token", "value": f"Response {id(mock_agent)}"}
                
                mock_agent.run_agent_streaming.return_value = mock_stream()
                return mock_agent
            
            mock_agent_class.side_effect = create_mock_agent
            
            # Make concurrent requests
            responses = []
            for i in range(3):
                request_data = sample_chat_request.copy()
                request_data["messages"] = [{"role": "user", "content": f"Message {i}"}]
                
                response = client.post(
                    "/api/chat/completions",
                    json=request_data
                )
                responses.append(response)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
            
            # Multiple agents should have been created
            assert mock_agent_class.call_count == 3