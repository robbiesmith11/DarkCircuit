import { useState, useEffect, useRef, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatContainer } from './components/ChatContainer';
import { ToolShortcuts } from './components/ToolShortcuts';
import { ToastContainer, toast } from 'react-toastify';
import SSHConnectForm from './components/SSHConnectForm';
import XTerminal from './components/XTerminal';
import 'react-toastify/dist/ReactToastify.css';
import appLogo from '../src/components/DarkCircuit_Logo_Blue_PNG.png';

const TERMINAL_WS_URL = import.meta.env.VITE_TERMINAL_WS_URL || '';
const API_BASE_URL = import.meta.env.VITE_BACKEND_API_URL || '';

function App() {
  // SSH connection state
  const [isSshConnected, setIsSshConnected] = useState(false);
  const [isSshConnecting, setIsSshConnecting] = useState(false);
  const [sshConnectionError, setSshConnectionError] = useState<string | null>(null);

  // Add state for target configuration
  const [selectedChallenge, setSelectedChallenge] = useState('');
  const [targetIp, setTargetIp] = useState('');

  // A ref to store the execute command function
  const executeCommandRef = useRef<((command: string, commandId?: number) => void) | null>(null);

  // Store pending agent commands
  const pendingCommandsRef = useRef<Map<number, { command: string, resolve: (output: string) => void }>>(new Map());

  const webSocketUrl = `${TERMINAL_WS_URL}/ws/ssh-terminal`;

  // Handle SSH connection
  const handleSshConnect = async (data: {
    host: string;
    port: number;
    username: string;
    password: string;
    keyPath?: string;
  }) => {
    setIsSshConnecting(true);
    setSshConnectionError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/ssh/connect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed with status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        setIsSshConnected(true);
        toast.success('SSH connected successfully');

        // If there's a warning in the response, show it
        if (result.warning) {
          toast.warning(result.warning);
        }
      } else {
        setSshConnectionError(result.error || 'Failed to connect to SSH server');
        toast.error(`SSH connection failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error connecting to SSH:', error);
      setSshConnectionError(error instanceof Error ? error.message : String(error));
      toast.error(`Error connecting to SSH: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSshConnecting(false);
    }
  };

  // Handle SSH disconnection - memoize to prevent recreation
  const handleSshDisconnect = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/ssh/disconnect`, {
        method: 'POST',
      });

      const data = await response.json();

      if (data.success) {
        setIsSshConnected(false);
        toast.success('SSH disconnected successfully');
      } else {
        toast.error(`Failed to disconnect SSH: ${data.error}`);
      }
    } catch (error) {
      console.error('Error disconnecting SSH:', error);
      toast.error(`Error disconnecting SSH: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [API_BASE_URL]);

  // Memoize the tool click handler
  const handleToolClick = useCallback((command: string) => {
    if (executeCommandRef.current) {
      executeCommandRef.current(command);
    } else {
      console.log(`Tool command requested, but terminal not ready: ${command}`);
      toast.warning('Terminal not ready. Please try again in a moment.');
    }
  }, []);

  // Function to handle SSH commands from the ChatContainer - memoize to prevent recreation
  const handleSshToolCall = useCallback(async (command: string, commandId?: number): Promise<string> => {
    if (isSshConnected && executeCommandRef.current) {
      if (commandId !== undefined) {
        return new Promise<string>((resolve) => {
          // Store this promise's resolve function to call when output is ready
          pendingCommandsRef.current.set(commandId, { command, resolve });

          // Execute the command in the terminal, passing the command ID
          executeCommandRef.current!(command, commandId);

          // Set a timeout to resolve the promise after some time if not resolved
          setTimeout(() => {
            // Check if this command is still pending
            if (pendingCommandsRef.current.has(commandId)) {
              const pendingCommand = pendingCommandsRef.current.get(commandId);
              if (pendingCommand) {
                pendingCommand.resolve(`Command timed out after 60 seconds. Check terminal for results.`);
                pendingCommandsRef.current.delete(commandId);
              }
            }
          }, 60000); // 60 seconds timeout
        });
      } else {
        // For commands without an ID, just execute without waiting for output
        executeCommandRef.current!(command);
        return "Command executed";
      }
    } else {
      // If not connected, show a toast warning
      toast.warning('Terminal not connected. Please connect SSH to execute commands.');
      return "Terminal not connected";
    }
  }, [isSshConnected]);

  // Handle terminal output - memoize to prevent recreation
  const handleTerminalOutput = useCallback((commandId: number, output: string, isPartial: boolean = false) => {
    console.log(`Received terminal output for command ${commandId} (${isPartial ? 'partial' : 'complete'}): ${output.substring(0, 100)}...`);

    // Send all outputs to the backend (both partial and complete)
    sendOutputToBackend(commandId, output, isPartial);

    // Only resolve the promise for complete responses
    // or if the partial output contains a completion indicator
    const pendingCommand = pendingCommandsRef.current.get(commandId);
    if (pendingCommand && (!isPartial || output.includes("[Command completed]"))) {
      pendingCommand.resolve(output);
      pendingCommandsRef.current.delete(commandId);
    }
  }, []);

  // Send output to backend API
  const sendOutputToBackend = async (commandId: number, output: string, isPartial: boolean = false) => {
    try {
      await fetch(`${API_BASE_URL}/api/terminal/output`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command_id: commandId,
          output: output,
          is_partial: isPartial
        }),
      });
    } catch (error) {
      console.error('Error sending terminal output to backend:', error);
    }
  };

  // Memoize the registerExecuteCommand function
  const registerExecuteCommand = useCallback((fn: (command: string, commandId?: number) => void) => {
    executeCommandRef.current = fn;
  }, []);

  // Memoize the challenge selection handler
  const handleChallengeSelect = useCallback((challenge: string) => {
    console.log("Selecting challenge:", challenge);
    setSelectedChallenge(challenge);
  }, []);

  // Memoize the target IP change handler
  const handleTargetIpChange = useCallback((ip: string) => {
    console.log("Updating target IP:", ip);
    setTargetIp(ip);
  }, []);

  // For debugging
  useEffect(() => {
    console.log("App state updated:", { selectedChallenge, targetIp });
  }, [selectedChallenge, targetIp]);

  return (
    <div className="flex h-screen bg-black">
      {/* Sidebar - Pass handlers */}
      <Sidebar
        selectedChallenge={selectedChallenge}
        targetIp={targetIp}
        onChallengeSelect={handleChallengeSelect}
        onTargetIpChange={handleTargetIpChange}
      />

      <div className="flex-1 p-4 overflow-auto">
        {/* Background logo */}
        <img
          src={appLogo}
          alt="Dark Circuit Logo Background"
          className="absolute top-1/2 left-1/2 w-[80%] max-w-4xl opacity-15 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-0"
        />
        <div className="grid grid-cols-2 gap-4 h-full">
          <div className="space-y-4 h-full">
            {/* Pass the SSH command handler and the actual state values */}
            <ChatContainer
              onSshToolCall={handleSshToolCall}
              selectedChallenge={selectedChallenge}
              targetIp={targetIp}
            />
          </div>
          <div className="flex flex-col space-y-4 h-full">
            {/* SSH Connection Form or Terminal */}
            <div className="flex-1 flex flex-col">
              {!isSshConnected ? (
                <div className="bg-gray-800 rounded-lg p-4 h-full flex flex-col justify-center">
                  <h1 className="font-bold text-white text-lg mb-6 text-center">Connect to Remote Terminal</h1>
                  <SSHConnectForm
                    onConnect={handleSshConnect}
                    isConnecting={isSshConnecting}
                    isConnected={isSshConnected}
                    connectionError={sshConnectionError}
                  />
                </div>
              ) : (
                <XTerminal
                  webSocketUrl={webSocketUrl}
                  isConnected={isSshConnected}
                  onDisconnect={handleSshDisconnect}
                  registerExecuteCommand={registerExecuteCommand}
                  onTerminalOutput={handleTerminalOutput}
                />
              )}
            </div>

            {/* Tool Shortcuts */}
            {isSshConnected && (
              <div>
                <ToolShortcuts onToolClick={handleToolClick} />
              </div>
            )}
          </div>
        </div>
      </div>
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
        aria-label="Notifications"
      />
    </div>
  );
}

export default App;