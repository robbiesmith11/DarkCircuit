"""
Unit tests for SSH connection management functionality.

Tests cover:
- SSH connection establishment (TC-SSH-001)
- Invalid credentials handling (TC-SSH-002) 
- Unreachable host handling (TC-SSH-003)
- Connection cleanup and state management
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import paramiko

pytestmark = [pytest.mark.unit, pytest.mark.ssh]

# Import the functions to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import components, skip tests if dependencies are missing
try:
    from local_app import _setup_ssh_connection, _close_ssh_connection, _strip_ansi_codes
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping SSH tests due to missing dependencies: {e}", allow_module_level=True)


class TestSSHConnectionManagement:
    """Test class for SSH connection management functions."""

    def test_successful_ssh_connection(self, mock_paramiko_client, valid_ssh_credentials):
        """
        Test Case TC-SSH-001: Successful SSH Connection (Normal Scenario)
        
        Tests that a valid SSH connection can be established successfully
        with proper credentials and that the connection state is updated correctly.
        """
        # Mock the global ssh_state
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None, 
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Mock paramiko.SSHClient
            with patch('local_app.paramiko.SSHClient', return_value=mock_paramiko_client):
                
                # Execute the connection
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                # Verify the result
                assert result["success"] is True
                assert "SSH connection established" in result["message"]
                assert valid_ssh_credentials["host"] in result["message"]
                
                # Verify mock calls
                mock_paramiko_client.set_missing_host_key_policy.assert_called_once()
                mock_paramiko_client.connect.assert_called_once()
                mock_paramiko_client.exec_command.assert_called_once_with("echo connected", timeout=5)
                
                # Verify connection state
                assert mock_state["connected"] is True
                assert mock_state["client"] == mock_paramiko_client
                assert mock_state["error"] is None

    def test_ssh_connection_invalid_credentials(self, valid_ssh_credentials):
        """
        Test Case TC-SSH-002: SSH Connection with Invalid Credentials (Edge Case)
        
        Tests that authentication failures are handled properly and return
        appropriate error messages without crashing the application.
        """
        # Mock the global ssh_state
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False, 
            "error": None,
            "running": False
        }) as mock_state:
            
            # Create mock client that raises AuthenticationException
            mock_client = Mock()
            mock_client.connect.side_effect = paramiko.ssh_exception.AuthenticationException("Auth failed")
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_client):
                
                # Execute connection with invalid credentials
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username="invalid_user",
                    password="wrong_password"
                )
                
                # Verify the result
                assert result["success"] is False
                assert "Authentication failed" in result["error"]
                assert "Please check your username, password or key" in result["error"]
                
                # Verify connection state
                assert mock_state["connected"] is False
                assert "Authentication failed" in mock_state["error"]

    def test_ssh_connection_unreachable_host(self, valid_ssh_credentials):
        """
        Test Case TC-SSH-003: SSH Connection to Unreachable Host (Edge Case)
        
        Tests that connection attempts to unreachable hosts timeout properly
        and return appropriate error messages.
        """
        # Mock the global ssh_state
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Create mock client that raises NoValidConnectionsError
            mock_client = Mock()
            # Create a proper error dict for NoValidConnectionsError
            error_dict = {('10.10.10.1', 22): Exception("Connection timed out")}
            mock_client.connect.side_effect = paramiko.ssh_exception.NoValidConnectionsError(error_dict)
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_client):
                
                # Execute connection to unreachable host
                result = _setup_ssh_connection(
                    host="192.168.999.999",  # Invalid IP
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                # Verify the result
                assert result["success"] is False
                assert "Could not connect" in result["error"]
                assert "192.168.999.999:22" in result["error"]
                assert "Server may be down or unreachable" in result["error"]
                
                # Verify connection state
                assert mock_state["connected"] is False
                assert "Could not connect" in mock_state["error"]

    def test_ssh_connection_with_key_authentication(self, valid_ssh_credentials):
        """
        Test SSH connection using key-based authentication.
        
        Tests that SSH connections can be established using private key files
        instead of password authentication.
        """
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Mock paramiko components
            mock_client = Mock()
            mock_key = Mock()
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_client), \
                 patch('local_app.paramiko.RSAKey.from_private_key_file', return_value=mock_key):
                
                # Configure successful connection
                mock_transport = Mock()
                mock_transport.is_active.return_value = True
                mock_client.get_transport.return_value = mock_transport
                
                stdout_mock = Mock()
                stdout_mock.read.return_value = b"connected"
                mock_client.exec_command.return_value = (Mock(), stdout_mock, Mock())
                
                # Execute connection with key
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password="key_passphrase",
                    key_path="/path/to/private/key"
                )
                
                # Verify the result
                assert result["success"] is True
                assert "SSH connection established" in result["message"]
                
                # Verify key-based auth was used
                mock_client.connect.assert_called_once()
                connect_args = mock_client.connect.call_args[1]
                assert "pkey" in connect_args
                assert connect_args["pkey"] == mock_key

    def test_ssh_connection_test_command_failure(self, mock_paramiko_client, valid_ssh_credentials):
        """
        Test SSH connection when test command fails.
        
        Tests that connections are rejected if the test command doesn't
        return the expected result, indicating a problematic connection.
        """
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Configure test command to fail
            stdout_mock = Mock()
            stdout_mock.read.return_value = b"unexpected_output"
            mock_paramiko_client.exec_command.return_value = (Mock(), stdout_mock, Mock())
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_paramiko_client):
                
                # Execute connection
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                # Verify the result
                assert result["success"] is False
                assert "Connection test failed" in result["error"]
                
                # Verify client was closed
                mock_paramiko_client.close.assert_called_once()

    def test_close_ssh_connection(self):
        """
        Test SSH connection cleanup functionality.
        
        Tests that closing SSH connections properly cleans up all resources
        including channels, clients, and resets the connection state.
        """
        # Create mock objects
        mock_client = Mock()
        mock_channel = Mock()
        mock_transport = Mock()
        
        mock_client.get_transport.return_value = mock_transport
        mock_transport.is_active.return_value = True
        
        # Set up initial state
        with patch('local_app.ssh_state', {
            "client": mock_client,
            "channel": mock_channel,
            "connected": True,
            "error": None,
            "running": True
        }) as mock_state:
            
            # Execute cleanup
            _close_ssh_connection()
            
            # Verify cleanup actions
            mock_channel.close.assert_called_once()
            mock_transport.close.assert_called_once()
            mock_client.close.assert_called_once()
            
            # Verify state reset
            assert mock_state["client"] is None
            assert mock_state["channel"] is None
            assert mock_state["connected"] is False
            assert mock_state["running"] is False
            assert mock_state["error"] is None

    def test_close_ssh_connection_with_exceptions(self):
        """
        Test SSH connection cleanup with exceptions.
        
        Tests that cleanup continues even if individual cleanup steps fail,
        ensuring no resource leaks occur.
        """
        # Create mock objects that raise exceptions
        mock_client = Mock()
        mock_channel = Mock()
        mock_transport = Mock()
        
        mock_channel.close.side_effect = Exception("Channel close failed")
        mock_transport.close.side_effect = Exception("Transport close failed")
        mock_client.close.side_effect = Exception("Client close failed")
        mock_client.get_transport.return_value = mock_transport
        mock_transport.is_active.return_value = True
        
        # Set up initial state
        with patch('local_app.ssh_state', {
            "client": mock_client,
            "channel": mock_channel,
            "connected": True,
            "error": None,
            "running": True
        }) as mock_state:
            
            # Execute cleanup - should not raise exceptions
            _close_ssh_connection()
            
            # Verify state reset despite exceptions
            assert mock_state["client"] is None
            assert mock_state["channel"] is None
            assert mock_state["connected"] is False
            assert mock_state["running"] is False
            assert mock_state["error"] is None


class TestANSICodeStripping:
    """Test class for ANSI escape sequence handling."""

    def test_strip_ansi_codes_basic(self, ansi_test_data):
        """
        Test ANSI escape sequence removal.
        
        Tests that ANSI escape sequences used for terminal formatting
        are properly removed to get clean text for processing.
        """
        for input_text, expected_output in ansi_test_data:
            result = _strip_ansi_codes(input_text)
            assert result == expected_output, f"Failed for input: {repr(input_text)}"

    def test_strip_ansi_codes_non_string(self):
        """
        Test ANSI stripping with non-string input.
        
        Tests that non-string inputs are returned unchanged without errors.
        """
        # Test various non-string types
        assert _strip_ansi_codes(None) is None
        assert _strip_ansi_codes(123) == 123
        assert _strip_ansi_codes([1, 2, 3]) == [1, 2, 3]
        assert _strip_ansi_codes({"key": "value"}) == {"key": "value"}

    def test_strip_ansi_codes_complex_sequences(self):
        """
        Test complex ANSI escape sequences.
        
        Tests handling of complex terminal control sequences that might
        appear in real terminal output.
        """
        # Complex sequences with multiple codes
        complex_input = "\x1b[1;31;40mBold Red on Black\x1b[0m\x1b[2J\x1b[H\x1b[?25h"
        expected = "Bold Red on Black"
        assert _strip_ansi_codes(complex_input) == expected
        
        # Nested sequences
        nested_input = "\x1b[1m\x1b[31mNested\x1b[0m\x1b[0m"
        expected = "Nested"
        assert _strip_ansi_codes(nested_input) == expected


class TestSSHConnectionEdgeCases:
    """Test edge cases and error conditions for SSH connections."""

    def test_ssh_connection_timeout_during_connect(self, valid_ssh_credentials):
        """
        Test SSH connection timeout during connection establishment.
        """
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Mock client that times out
            mock_client = Mock()
            mock_client.connect.side_effect = TimeoutError("Connection timed out")
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_client):
                
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                assert result["success"] is False
                assert "Failed to establish SSH connection" in result["error"]
                assert mock_state["connected"] is False

    def test_ssh_connection_transport_failure(self, mock_paramiko_client, valid_ssh_credentials):
        """
        Test SSH connection when transport layer fails.
        """
        with patch('local_app.ssh_state', {
            "client": None,
            "channel": None,
            "connected": False,
            "error": None,
            "running": False
        }) as mock_state:
            
            # Configure transport to return None
            mock_paramiko_client.get_transport.return_value = None
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_paramiko_client):
                
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                # Should still succeed as transport check is after connection
                assert result["success"] is True

    def test_ssh_connection_existing_connection_cleanup(self, mock_paramiko_client, valid_ssh_credentials):
        """
        Test that existing connections are properly cleaned up before new connections.
        """
        # Mock existing connection
        old_client = Mock()
        old_channel = Mock()
        
        with patch('local_app.ssh_state', {
            "client": old_client,
            "channel": old_channel,
            "connected": True,
            "error": None,
            "running": True
        }) as mock_state:
            
            with patch('local_app.paramiko.SSHClient', return_value=mock_paramiko_client), \
                 patch('local_app._close_ssh_connection') as mock_close:
                
                result = _setup_ssh_connection(
                    host=valid_ssh_credentials["host"],
                    port=valid_ssh_credentials["port"],
                    username=valid_ssh_credentials["username"],
                    password=valid_ssh_credentials["password"]
                )
                
                # Verify old connection was closed
                mock_close.assert_called_once()
                assert result["success"] is True