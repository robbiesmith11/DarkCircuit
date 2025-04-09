import React, { useEffect, useRef, useState } from 'react';
import { Terminal as TerminalIcon, Maximize2, Minimize2, X, RotateCcw } from 'lucide-react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import 'xterm/css/xterm.css';

interface XTerminalProps {
  webSocketUrl: string;
  isConnected: boolean;
  onDisconnect: () => void;
  registerExecuteCommand?: (fn: (command: string) => void) => void;
}

export const XTerminal: React.FC<XTerminalProps> = ({
  webSocketUrl,
  isConnected,
  onDisconnect,
  registerExecuteCommand
}) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [terminalReady, setTerminalReady] = useState(false);
  // Add a reconnection trigger state
  const [reconnectTrigger, setReconnectTrigger] = useState(0);
  // Add a flag to track if disconnection was unexpected
  const unexpectedDisconnectRef = useRef(false);
  // Track auto-reconnect attempts
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 3;

  // Initialize the terminal
  useEffect(() => {
    if (!terminalRef.current) return;

    // Create terminal only once
    if (!xtermRef.current) {
      // Create and configure terminal
      const terminal = new Terminal({
        cursorBlink: true,
        theme: {
          background: '#0f0f0f',
          foreground: '#00FF00',
          cursor: '#00FF00',
          black: '#000000',
          red: '#C51E14',
          green: '#1DC121',
          yellow: '#C7C329',
          blue: '#0A2FC4',
          magenta: '#C839C5',
          cyan: '#20C5C6',
          white: '#C7C7C7',
          brightBlack: '#686868',
          brightRed: '#FD6F6B',
          brightGreen: '#67F86F',
          brightYellow: '#FFFA72',
          brightBlue: '#6A76FB',
          brightMagenta: '#FD7CFC',
          brightCyan: '#68FDFE',
          brightWhite: '#FFFFFF'
        },
        fontFamily: '"Cascadia Code", Menlo, monospace',
        fontSize: 14,
        lineHeight: 1.2,
        scrollback: 1000,
        allowTransparency: true
      });

      // Create fit addon
      const fitAddon = new FitAddon();
      terminal.loadAddon(fitAddon);
      fitAddonRef.current = fitAddon;

      // Add web links addon
      terminal.loadAddon(new WebLinksAddon());

      // Store the terminal
      xtermRef.current = terminal;

      // Open the terminal
      terminal.open(terminalRef.current);

      // Fit to container
      setTimeout(() => {
        if (fitAddonRef.current) {
          fitAddonRef.current.fit();
        }
      }, 100);

      // Handle terminal input
      terminal.onData((data) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(data);
        } else if (status !== 'connecting') {
          // If not connected and not already attempting to connect, show feedback
          terminal.writeln('\r\n\x1b[1;31mNot connected. Attempting to reconnect...\x1b[0m');
          // Try to reconnect
          setReconnectTrigger(prev => prev + 1);
        }
      });

      // Set terminal ready
      setTerminalReady(true);

      // Register command execution function if needed
      if (registerExecuteCommand) {
        registerExecuteCommand((command: string) => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            // Send the command with a newline to execute it
            wsRef.current.send(command + '\n');

            // Also echo the command to the terminal for user feedback
            terminal.writeln(`$ ${command}`);
          } else if (status !== 'connecting') {
            // If not connected and not already attempting to connect, show feedback
            terminal.writeln('\r\n\x1b[1;31mNot connected. Attempting to reconnect...\x1b[0m');
            // Try to reconnect
            setReconnectTrigger(prev => prev + 1);
          }
        });
      }
    }

    // Clean up on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      if (registerExecuteCommand) {
        registerExecuteCommand(() => {});
      }
    };
  }, [registerExecuteCommand]);

  // Resize handling
  useEffect(() => {
    const handleResize = () => {
      if (fitAddonRef.current) {
        fitAddonRef.current.fit();
      }
    };

    window.addEventListener('resize', handleResize);

    // Also resize when maximized state changes
    if (terminalReady) {
      setTimeout(handleResize, 100);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [isMaximized, terminalReady]);

  // Create a dedicated websocket connection function
  const connectWebSocket = () => {
    // Reset reconnect attempts if this is a manual reconnection
    if (reconnectTrigger > 0) {
      reconnectAttemptsRef.current = 0;
    }

    // Close existing connection if any
    if (wsRef.current) {
      // Only attempt to close if not already closed
      if (wsRef.current.readyState !== WebSocket.CLOSED) {
        unexpectedDisconnectRef.current = false; // Mark as expected disconnection
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    // Don't proceed if we shouldn't be connected
    if (!isConnected || !webSocketUrl || !terminalReady) {
      return;
    }

    setStatus('connecting');

    // Create new WebSocket
    const ws = new WebSocket(webSocketUrl);

    ws.onopen = () => {
      setStatus('connected');
      reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
      if (xtermRef.current) {
        xtermRef.current.writeln('\r\n\x1b[1;32mConnected to remote terminal.\x1b[0m\r\n');
      }
    };

    ws.onmessage = (event) => {
      if (xtermRef.current) {
        // Handle both text and binary messages
        if (typeof event.data === 'string') {
          xtermRef.current.write(event.data);
        } else {
          // Handle binary data (for proper terminal escape sequences)
          const reader = new FileReader();
          reader.onload = () => {
            if (reader.result && xtermRef.current) {
              xtermRef.current.write(new Uint8Array(reader.result as ArrayBuffer));
            }
          };
          reader.readAsArrayBuffer(event.data);
        }
      }
    };

    ws.onclose = (event) => {
      // Mark this as unexpected disconnection if it wasn't a manual close
      const wasUnexpected = !event.wasClean;
      unexpectedDisconnectRef.current = wasUnexpected;

      setStatus('disconnected');

      if (xtermRef.current) {
        if (!wasUnexpected) {
          xtermRef.current.writeln('\r\n\x1b[1;33mDisconnected from terminal.\x1b[0m');
        } else {
          xtermRef.current.writeln('\r\n\x1b[1;31mConnection lost unexpectedly.\x1b[0m');

          // Auto-reconnect for unexpected disconnections, with limits
          if (isConnected && reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current++;
            xtermRef.current.writeln(`\r\n\x1b[1;33mAttempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...\x1b[0m`);

            // Use incremental backoff for reconnection attempts
            const backoffDelay = Math.min(1000 * Math.pow(1.5, reconnectAttemptsRef.current - 1), 8000);

            setTimeout(() => {
              setReconnectTrigger(prev => prev + 1);
            }, backoffDelay);
          } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
            xtermRef.current.writeln('\r\n\x1b[1;31mMaximum reconnection attempts reached. Please try manual reconnect.\x1b[0m');
          }
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      // Don't set disconnected here, let onclose handle it
      if (xtermRef.current) {
        xtermRef.current.writeln('\r\n\x1b[1;31mWebSocket error occurred.\x1b[0m');
      }
    };

    wsRef.current = ws;
  };

  // Connect to WebSocket when ready and connected
  useEffect(() => {
    connectWebSocket();

    // Return cleanup function
    return () => {
      if (wsRef.current) {
        unexpectedDisconnectRef.current = false; // Mark as expected disconnection
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [webSocketUrl, isConnected, terminalReady, reconnectTrigger]);

  const disconnectTerminal = () => {
    unexpectedDisconnectRef.current = false; // Mark as expected disconnection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    onDisconnect();
  };

  const toggleMaximize = () => {
    setIsMaximized(!isMaximized);
  };

  const reconnectTerminal = () => {
    if (xtermRef.current) {
      xtermRef.current.writeln('\r\n\x1b[1;33mManually reconnecting...\x1b[0m');
    }

    // Reset reconnect attempts counter for manual reconnection
    reconnectAttemptsRef.current = 0;

    // Increment the reconnect trigger to force the useEffect to run again
    setReconnectTrigger(prev => prev + 1);
  };

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden border border-gray-700 flex flex-col ${
      isMaximized 
        ? 'fixed top-0 left-0 w-full h-full z-50' 
        : 'h-full'
    }`}>
      {/* Terminal header */}
      <div className="flex justify-between items-center bg-gray-900 px-4 py-2">
        <div className="flex items-center">
          <TerminalIcon size={18} className="text-purple-500 mr-2" />
          <span className="text-sm font-mono text-purple-500">
            Remote Terminal
            {status === 'connected' && <span className="text-green-500 ml-2">(connected)</span>}
            {status === 'connecting' && <span className="text-yellow-500 ml-2">(connecting...)</span>}
            {status === 'disconnected' && <span className="text-red-500 ml-2">(disconnected)</span>}
          </span>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={reconnectTerminal}
            className="text-gray-400 hover:text-blue-500 focus:outline-none"
            aria-label="Reconnect"
            title="Reconnect terminal"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={toggleMaximize}
            className="text-gray-400 hover:text-white focus:outline-none"
            aria-label={isMaximized ? 'Minimize' : 'Maximize'}
            title={isMaximized ? 'Minimize terminal' : 'Maximize terminal'}
          >
            {isMaximized ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button
            onClick={disconnectTerminal}
            className="text-gray-400 hover:text-red-500 focus:outline-none"
            aria-label="Disconnect"
            title="Disconnect terminal"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Terminal container */}
      <div
        ref={terminalRef}
        className="flex-1 overflow-hidden"
        style={{
          padding: '4px',
          backgroundColor: '#0f0f0f'
        }}
      />
    </div>
  );
};

export default XTerminal;