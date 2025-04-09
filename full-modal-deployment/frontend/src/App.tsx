import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatContainer } from './components/ChatContainer';
import { ToolShortcuts } from './components/ToolShortcuts';
import { ToastContainer, toast } from 'react-toastify';
import SSHConnectForm from './components/SSHConnectForm';
import XTerminal from './components/XTerminal';
import 'react-toastify/dist/ReactToastify.css';

const TERMINAL_WS_URL = import.meta.env.VITE_TERMINAL_WS_URL || '';
const API_BASE_URL = import.meta.env.VITE_BACKEND_API_URL || '';

function App() {
  // SSH connection state
  const [isSshConnected, setIsSshConnected] = useState(false);
  const [isSshConnecting, setIsSshConnecting] = useState(false);
  const [sshConnectionError, setSshConnectionError] = useState<string | null>(null);

  const executeCommandRef = useRef<((command: string) => void) | null>(null);

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

  // Handle SSH disconnection
  const handleSshDisconnect = async () => {
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
  };

  const handleToolClick = (command: string) => {
    if (executeCommandRef.current) {
      executeCommandRef.current(command);
    } else {
      console.log(`Tool command requested, but terminal not ready: ${command}`);
      toast.warning('Terminal not ready. Please try again in a moment.');
    }
  };

  const registerExecuteCommand = (fn: (command: string) => void) => {
    executeCommandRef.current = fn;
  };

  const handleServiceToggle = (service: string) => {
    if (isSshConnected && executeCommandRef.current) {
      let command = '';

      switch (service) {
        case 'apache':
          command = 'sudo service apache2 restart';
          break;
        case 'squid':
          command = 'sudo service squid restart';
          break;
        default:
          toast.warning(`Unknown service: ${service}`);
          return;
      }

      toast.info(`Toggling ${service} service...`);
      executeCommandRef.current(command);
    } else {
      toast.warning('Please connect to SSH first before managing services.');
    }
  };

  return (
    <div className="flex h-screen bg-black">
      <Sidebar
        onServiceToggle={handleServiceToggle}
      />
      <div className="flex-1 p-4 overflow-auto">
        <div className="grid grid-cols-2 gap-4 h-full">
          <div className="space-y-4 h-full">
            <ChatContainer />
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