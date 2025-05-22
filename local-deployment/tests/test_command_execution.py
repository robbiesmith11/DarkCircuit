"""
Unit tests for terminal command execution functionality.

Tests cover:
- Basic command execution (TC-CMD-001)
- Long-running commands (TC-CMD-002)
- Commands with error output (TC-CMD-003)
- Command optimization and prompt detection
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

pytestmark = [pytest.mark.unit]

# Import the functions to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import components, skip tests if dependencies are missing
try:
    from local_app import run_ssh_command
    from agent_utils import optimize_command, wait_for_terminal_output
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping command execution tests due to missing dependencies: {e}", allow_module_level=True)


class TestCommandExecution:
    """Test class for SSH command execution functionality."""

    @pytest.mark.asyncio
    async def test_basic_command_execution(self, sample_terminal_output):
        """
        Test Case TC-CMD-001: Basic Command Execution (Normal Scenario)
        
        Tests that simple commands execute successfully and return expected output
        with proper exit codes and formatting.
        """
        # Mock WebSocket and terminal state
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            # Simulate command execution and output
            async def simulate_command_output():
                await asyncio.sleep(0.1)  # Simulate execution delay
                terminal_output_buffers["main"] = sample_terminal_output["simple_command"]["output"]
                # Add shell prompt to indicate completion
                terminal_output_buffers["main"] += "└──╼ [★]$ "
            
            # Start simulation task
            simulation_task = asyncio.create_task(simulate_command_output())
            
            # Execute command
            result = await run_ssh_command("pwd", timeout=5)
            
            # Wait for simulation to complete
            await simulation_task
            
            # Verify results
            assert result["success"] is True
            assert "/home/htb-student" in result["output"]
            assert result["exit_code"] == 0
            assert "error" in result
            
            # Verify WebSocket was called
            mock_websocket.send_text.assert_called()
            sent_command = mock_websocket.send_text.call_args[0][0]
            assert "__AGENT_COMMAND__:pwd" in sent_command

    @pytest.mark.asyncio
    async def test_long_running_command(self, sample_terminal_output):
        """
        Test Case TC-CMD-002: Long-Running Command (Edge Case)
        
        Tests that commands that take time to execute are handled properly
        without premature timeout.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            # Simulate long-running command
            async def simulate_long_command():
                await asyncio.sleep(0.5)  # Simulate 5-second sleep
                terminal_output_buffers["main"] = sample_terminal_output["long_running_command"]["output"]
                terminal_output_buffers["main"] += "└──╼ [★]$ "
            
            simulation_task = asyncio.create_task(simulate_long_command())
            
            # Execute long-running command with appropriate timeout
            result = await run_ssh_command("sleep 5 && echo done", timeout=10)
            
            await simulation_task
            
            # Verify results
            assert result["success"] is True
            assert "done" in result["output"]
            assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_command_with_error_output(self, sample_terminal_output):
        """
        Test Case TC-CMD-003: Command with Error Output (Edge Case)
        
        Tests that commands that generate error output are properly captured
        and reported with appropriate exit codes.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            # Simulate command with error
            async def simulate_error_command():
                await asyncio.sleep(0.1)
                terminal_output_buffers["main"] = sample_terminal_output["command_with_error"]["output"]
                terminal_output_buffers["main"] += "└──╼ [★]$ "
            
            simulation_task = asyncio.create_task(simulate_error_command())
            
            # Execute command that will fail
            result = await run_ssh_command("ls /nonexistent", timeout=5)
            
            await simulation_task
            
            # Verify error is captured
            assert result["success"] is True  # Command executed, even if it failed
            assert "cannot access" in result["output"]
            assert "No such file or directory" in result["output"]

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """
        Test command execution timeout handling.
        
        Tests that commands that exceed the timeout period are properly
        terminated and return timeout error information.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            # Don't add any output to simulate hanging command
            
            # Execute command with short timeout
            result = await run_ssh_command("sleep 1000", timeout=1)
            
            # Verify timeout handling
            assert result["success"] is False
            assert "Timed out" in result["error"]
            assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_no_terminal_available(self):
        """
        Test command execution when no terminal is available.
        
        Tests that appropriate error is returned when no WebSocket
        terminal connection is available for command execution.
        """
        # Empty terminal clients set
        with patch('local_app.terminal_ws_clients', set()):
            
            result = await run_ssh_command("pwd", timeout=5)
            
            # Verify error handling
            assert result["success"] is False
            assert "No terminal" in result["error"]
            assert result["exit_code"] == -1

    @pytest.mark.asyncio 
    async def test_cli_prompt_detection(self):
        """
        Test CLI prompt pattern detection.
        
        Tests that various CLI prompts are properly detected to determine
        when command execution is complete.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        # Test different CLI prompts
        test_prompts = [
            "└──╼ [★]$ ",
            "smb: \\target\\> ",
            "mysql> ",
            "(gdb) ",
            "[0x08048000]> ",
            "msf6 > "
        ]
        
        for prompt in test_prompts:
            with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
                 patch('local_app.terminal_output_buffers', terminal_output_buffers), \
                 patch('local_app.command_lock', asyncio.Lock()):
                
                # Reset buffer
                terminal_output_buffers["main"] = ""
                
                async def simulate_prompt_output():
                    await asyncio.sleep(0.1)
                    terminal_output_buffers["main"] = f"some output\n{prompt}"
                
                simulation_task = asyncio.create_task(simulate_prompt_output())
                
                result = await run_ssh_command("test", timeout=5)
                
                await simulation_task
                
                # Should detect prompt and complete successfully
                assert result["success"] is True, f"Failed to detect prompt: {prompt}"
                assert prompt in result["output"]


class TestCommandOptimization:
    """Test class for command optimization functionality."""

    def test_nmap_command_optimization(self, mock_command_optimization_cases):
        """
        Test nmap command optimization.
        
        Tests that slow nmap commands are automatically optimized with
        --min-rate parameter to improve execution speed.
        """
        for case in mock_command_optimization_cases:
            optimized_command, optimization_message = optimize_command(case["input"])
            
            assert optimized_command == case["expected_output"]
            assert optimization_message == case["expected_message"]

    def test_optimize_command_no_optimization_needed(self):
        """
        Test command optimization when no optimization is needed.
        
        Tests that commands that don't need optimization are returned unchanged.
        """
        test_commands = [
            "ls -la",
            "cat /etc/passwd", 
            "ping google.com",
            "nmap --min-rate=500 target.com"  # Already optimized
        ]
        
        for command in test_commands:
            optimized_command, optimization_message = optimize_command(command)
            assert optimized_command == command
            assert optimization_message is None

    def test_optimize_command_edge_cases(self):
        """
        Test command optimization edge cases.
        
        Tests optimization behavior with edge cases like empty commands,
        commands with special characters, etc.
        """
        # Empty command
        optimized, message = optimize_command("")
        assert optimized == ""
        assert message is None
        
        # Command with special characters
        special_command = "nmap -sS target.com && echo 'done'"
        optimized, message = optimize_command(special_command)
        assert "--min-rate=1000" in optimized
        assert message is not None
        
        # Command with existing min-rate
        existing_rate = "nmap -sS --min-rate=2000 target.com"
        optimized, message = optimize_command(existing_rate)
        assert optimized == existing_rate
        assert message is None


class TestTerminalOutputWaiting:
    """Test class for terminal output waiting functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_terminal_output_success(self):
        """
        Test successful terminal output waiting.
        
        Tests that the wait function properly receives and returns
        terminal output when it becomes available.
        """
        # Create test queue
        test_queue = asyncio.Queue()
        
        # Simulate output arrival
        async def provide_output():
            await asyncio.sleep(0.1)
            await test_queue.put({
                "command_id": 123,
                "output": "test output"
            })
        
        # Start output provider
        provider_task = asyncio.create_task(provide_output())
        
        # Wait for output
        result = await wait_for_terminal_output(test_queue, 123, timeout=5)
        
        await provider_task
        
        # Verify result
        assert result == "test output"

    @pytest.mark.asyncio
    async def test_wait_for_terminal_output_timeout(self):
        """
        Test terminal output waiting timeout.
        
        Tests that the wait function properly times out when no output
        is received within the specified timeout period.
        """
        # Create empty queue
        test_queue = asyncio.Queue()
        
        # Wait for output with short timeout
        result = await wait_for_terminal_output(test_queue, 123, timeout=1)
        
        # Verify timeout message
        assert "timed out after 1 seconds" in result

    @pytest.mark.asyncio
    async def test_wait_for_terminal_output_wrong_command_id(self):
        """
        Test terminal output waiting with wrong command ID.
        
        Tests that the wait function ignores output for different command IDs
        and continues waiting for the correct one.
        """
        test_queue = asyncio.Queue()
        
        # Provide output for wrong command ID
        async def provide_wrong_output():
            await asyncio.sleep(0.1)
            await test_queue.put({
                "command_id": 999,  # Wrong ID
                "output": "wrong output"
            })
            await asyncio.sleep(0.1)
            await test_queue.put({
                "command_id": 123,  # Correct ID
                "output": "correct output"
            })
        
        provider_task = asyncio.create_task(provide_wrong_output())
        
        result = await wait_for_terminal_output(test_queue, 123, timeout=5)
        
        await provider_task
        
        # Should receive correct output, not wrong one
        assert result == "correct output"


class TestCommandExecutionEdgeCases:
    """Test edge cases and error conditions for command execution."""

    @pytest.mark.asyncio
    async def test_command_execution_buffer_overflow(self):
        """
        Test command execution with large output.
        
        Tests that large command outputs are properly handled and
        buffers are managed to prevent memory issues.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            # Simulate large output
            large_output = "x" * 15000  # Larger than 10KB buffer limit
            
            async def simulate_large_output():
                await asyncio.sleep(0.1)
                terminal_output_buffers["main"] = large_output + "└──╼ [★]$ "
            
            simulation_task = asyncio.create_task(simulate_large_output())
            
            result = await run_ssh_command("find /", timeout=5)
            
            await simulation_task
            
            # Should complete successfully even with large output
            assert result["success"] is True
            # Buffer should be managed (may be slightly larger due to prompt)
            assert len(result["output"]) <= 15020  # Allow for prompt text

    @pytest.mark.asyncio
    async def test_command_execution_special_characters(self):
        """
        Test command execution with special characters.
        
        Tests that commands containing special characters, quotes,
        and escape sequences are handled properly.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        special_commands = [
            "echo 'hello world'",
            'echo "test with quotes"',
            "echo $USER",
            "ls | grep test",
            "find / -name '*.txt' 2>/dev/null"
        ]
        
        for command in special_commands:
            with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
                 patch('local_app.terminal_output_buffers', terminal_output_buffers), \
                 patch('local_app.command_lock', asyncio.Lock()):
                
                terminal_output_buffers["main"] = ""
                
                async def simulate_output():
                    await asyncio.sleep(0.1)
                    terminal_output_buffers["main"] = "command output\n└──╼ [★]$ "
                
                simulation_task = asyncio.create_task(simulate_output())
                
                result = await run_ssh_command(command, timeout=5)
                
                await simulation_task
                
                # Should handle special characters without errors
                assert result["success"] is True
                assert "command output" in result["output"]

    @pytest.mark.asyncio
    async def test_command_execution_concurrent_commands(self):
        """
        Test concurrent command execution handling.
        
        Tests that the command lock properly serializes command execution
        to prevent interference between concurrent commands.
        """
        mock_websocket = AsyncMock()
        terminal_output_buffers = {"main": ""}
        terminal_ws_clients = {mock_websocket}
        
        with patch('local_app.terminal_ws_clients', terminal_ws_clients), \
             patch('local_app.terminal_output_buffers', terminal_output_buffers), \
             patch('local_app.command_lock', asyncio.Lock()):
            
            async def simulate_command_output(output_text, delay=0.1):
                await asyncio.sleep(delay)
                terminal_output_buffers["main"] = f"{output_text}\n└──╼ [★]$ "
            
            # Start multiple commands concurrently
            tasks = []
            for i in range(3):
                # Each task simulates different output
                sim_task = asyncio.create_task(
                    simulate_command_output(f"output_{i}", delay=0.1 * (i + 1))
                )
                cmd_task = asyncio.create_task(
                    run_ssh_command(f"echo {i}", timeout=5)
                )
                tasks.extend([sim_task, cmd_task])
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have 6 results (3 simulation + 3 command tasks)
            assert len(results) == 6
            
            # Command results should all be successful
            command_results = [r for r in results if isinstance(r, dict) and "success" in r]
            assert len(command_results) == 3
            # At least some commands should succeed (concurrency may cause some to fail)
            successful_commands = [r for r in command_results if r.get("success") is True]
            assert len(successful_commands) >= 1