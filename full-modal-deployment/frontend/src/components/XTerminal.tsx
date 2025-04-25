import React, { useEffect, useRef, useState } from 'react';
import { Terminal as TerminalIcon, Maximize2, Minimize2, X, RotateCcw, AlertCircle } from 'lucide-react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import 'xterm/css/xterm.css';

interface XTerminalProps {
  webSocketUrl: string;
  isConnected: boolean;
  onDisconnect: () => void;
  registerExecuteCommand?: (fn: (command: string, commandId?: number) => void) => void;
  // Add new props for agent command execution
  onTerminalOutput?: (commandId: number, output: string, isPartial?: boolean) => void;
}

export const XTerminal: React.FC<XTerminalProps> = ({
  webSocketUrl,
  isConnected,
  onDisconnect,
  registerExecuteCommand,
  onTerminalOutput
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
  // Add buffer for output collection
  const outputBufferRef = useRef<string>("");
  // Track active commands with simplified structure
  const activeCommandsRef = useRef<Map<number, {
    command: string,
    timestamp: number,
    outputBuffer: string,
    updateTimeout: NodeJS.Timeout | null,
    lastOutputLength: number
  }>>(new Map());
  // Add a stable connection state reference to prevent race conditions
  const connectionStateRef = useRef<{
    isWebSocketClosing: boolean;
    reconnectTimer: NodeJS.Timeout | null;
  }>({
    isWebSocketClosing: false,
    reconnectTimer: null,
  });

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
        registerExecuteCommand((command: string, commandId?: number) => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            // Clear output buffer before executing a new command
            outputBufferRef.current = "";

            // Send the command with a newline to execute it
            wsRef.current.send(command + '\n');

            // If this is an agent command with ID, track it
            if (commandId !== undefined) {
              console.log(`Starting command tracking for ID ${commandId}: ${command}`);

              // Set up output collection for this command
              startOutputCollection(commandId, command);
            }
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

      // Clear any active timeouts - simplified with single timeout reference
      activeCommandsRef.current.forEach(cmd => {
        if (cmd.updateTimeout) {
          clearTimeout(cmd.updateTimeout);
        }
      });
    };
  }, [registerExecuteCommand]);

  // Unified function to start output collection for a command
  const startOutputCollection = (commandId: number, command: string) => {
    // Initialize output buffer for this command
    let commandOutputBuffer = "";

    // Detect long-running commands
    const isLongRunning = /nmap|enum4linux|gobuster|dirbuster|nikto|hydra|masscan|wpscan/.test(command);

    // Configure timing parameters - use more frequent updates for all commands
    const pollingInterval = isLongRunning ? 1500 : 2500; // Poll more frequently for long-running commands
    const maxLifetime = 120000; // 2 minutes maximum runtime

    // Track state
    let lastOutputLength = 0;
    const startTime = Date.now();

    // Improved shell prompt pattern with more variants
    const shellPromptPattern = /(\$ |# |> |sh-[0-9]+\.[0-9]+\$ |bash-[0-9]+\.[0-9]+\$ |\n[^\n]+[$#>]\s*$)/;

    // Unified function to process output
    const processOutput = () => {
      const cmd = activeCommandsRef.current.get(commandId);
      if (!cmd) return;

      // Check for timeout first
      //const elapsed = Date.now() - startTime;
      //if (elapsed > maxLifetime) {
        //onTerminalOutput?.(commandId, commandOutputBuffer + "\n\n[Command timed out after 2 minutes]", false);
        // Type-safe clearTimeout - only call if not null
        //if (cmd.updateTimeout) {
          //clearTimeout(cmd.updateTimeout);
        //}
        //activeCommandsRef.current.delete(commandId);
        //return;
      //}

      // Process any new output
      const latestOutput = outputBufferRef.current;
      if (latestOutput.length > 0) {
        commandOutputBuffer += latestOutput;
        outputBufferRef.current = ""; // Clear after consuming
      }

      // Check for completion
      const isComplete = shellPromptPattern.test(commandOutputBuffer);

      // Send updates if we have new content or if command completed
      if (commandOutputBuffer.length > lastOutputLength || isComplete) {
        const output = isComplete
          ? commandOutputBuffer + "\n\n[Command completed]"
          : commandOutputBuffer + "\n\n[Command still running... This is a partial output.]";

        onTerminalOutput?.(commandId, output, !isComplete);
        lastOutputLength = commandOutputBuffer.length;

        activeCommandsRef.current.set(commandId, {
          ...cmd,
          outputBuffer: commandOutputBuffer,
          lastOutputLength: lastOutputLength
        });
      }

      // Clean up if complete
      if (isComplete) {
        console.log(`Command ${commandId} completed after ${Math.round(elapsed/1000)}s`);
        // Type-safe clearTimeout
        if (cmd.updateTimeout) {
          clearTimeout(cmd.updateTimeout);
        }
        activeCommandsRef.current.delete(commandId);
        return;
      }

      // Schedule next check
      const nextTimeout = setTimeout(processOutput, pollingInterval);
      activeCommandsRef.current.set(commandId, {
        ...cmd,
        updateTimeout: nextTimeout
      });
    };

    // Start the processing loop
    const initialTimeout = setTimeout(processOutput, 500); // Start quickly

    // Register command in tracking map
    activeCommandsRef.current.set(commandId, {
      command,
      timestamp: startTime,
      outputBuffer: "",
      updateTimeout: initialTimeout,
      lastOutputLength: 0
    });

    console.log(`Started unified output collection for command ${commandId}: ${command}`);
  };

  // Improved handleTerminalOutput function
  const handleTerminalOutput = (data: string | ArrayBuffer | Blob | unknown) => {
    try {
      let outputStr = "";
      if (typeof data === 'string') {
        outputStr = data;
      } else if (data instanceof ArrayBuffer) {
        outputStr = new TextDecoder('utf-8').decode(data);
      } else if (typeof Blob !== 'undefined' && Object.prototype.toString.call(data) === '[object Blob]') {
        const blobData = data as Blob;
        const reader = new FileReader();
        reader.onload = () => {
          if (reader.result) {
            const resultStr = typeof reader.result === 'string'
              ? reader.result
              : new TextDecoder('utf-8').decode(reader.result as ArrayBuffer);
            handleTerminalOutput(resultStr); // recursive call
          }
        };
        reader.readAsText(blobData);
        return;
      } else {
        outputStr = String(data || '');
      }

      // Skip special messages
      if (outputStr.startsWith('__OUTPUT__')) {
        return;
      }

      // Accumulate output
      outputBufferRef.current += outputStr;

      // Check if any commands are actively using this output
      if (activeCommandsRef.current.size > 0) {
        // Log the buffer size occasionally to help with debugging
        if (outputBufferRef.current.length > 1000 && outputBufferRef.current.length % 5000 === 0) {
          console.log(`Output buffer size: ${outputBufferRef.current.length} bytes`);
        }
      }

      // Trim buffer to prevent memory issues
      if (outputBufferRef.current.length > 250000) {
        outputBufferRef.current = outputBufferRef.current.slice(-200000);
      }
    } catch (error) {
      console.error('Error processing terminal output:', error);
    }
  };

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

  // Improved WebSocket connection management
  const connectWebSocket = () => {
    // Clear any pending reconnect timer
    if (connectionStateRef.current.reconnectTimer) {
      clearTimeout(connectionStateRef.current.reconnectTimer);
      connectionStateRef.current.reconnectTimer = null;
    }

    // Don't reconnect if we're in the process of closing
    if (connectionStateRef.current.isWebSocketClosing) {
      console.log('Skipping connection attempt: WebSocket is currently closing');
      return;
    }

    // Don't attempt to connect if we're already connected or connecting
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.CONNECTING) {
        console.log('WebSocket already connecting, skipping reconnect');
        return;
      }
      if (wsRef.current.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected, skipping reconnect');
        return;
      }
    }

    // Reset reconnect attempts if this is a manual reconnection
    if (reconnectTrigger > 0) {
      reconnectAttemptsRef.current = 0;
    }

    // Close existing connection if any
    if (wsRef.current) {
      try {
        // Only attempt to close if not already closed or closing
        if (wsRef.current.readyState !== WebSocket.CLOSED && wsRef.current.readyState !== WebSocket.CLOSING) {
          console.log('Closing existing WebSocket before creating new connection');
          connectionStateRef.current.isWebSocketClosing = true;
          unexpectedDisconnectRef.current = false; // Mark as expected disconnection
          wsRef.current.close();

          // Reset the closing flag after a timeout to prevent deadlocks
          setTimeout(() => {
            connectionStateRef.current.isWebSocketClosing = false;
          }, 500);
        }
      } catch (err) {
        console.error('Error closing WebSocket:', err);
      }
      wsRef.current = null;
    }

    // Don't proceed if we shouldn't be connected
    if (!isConnected || !webSocketUrl || !terminalReady) {
      setStatus('disconnected');
      return;
    }

    console.log('Connecting to WebSocket:', webSocketUrl);
    setStatus('connecting');

    // Create new WebSocket with specific event handlers
    let ws: WebSocket;

    try {
      ws = new WebSocket(webSocketUrl);
    } catch (err) {
      console.error('Error creating WebSocket:', err);
      setStatus('disconnected');
      return;
    }

    // Store the new WebSocket immediately
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected successfully');
      setStatus('connected');
      reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
      connectionStateRef.current.isWebSocketClosing = false;

      if (xtermRef.current) {
        xtermRef.current.writeln('\r\n\x1b[1;32mConnected to remote terminal.\x1b[0m\r\n');
      }
    };

    ws.onmessage = (event) => {
      // Process all terminal output for command tracking
      handleTerminalOutput(event.data);

      // Handle regular terminal display
      if (xtermRef.current) {
        // Handle both text and binary messages
        if (typeof event.data === 'string') {
          // Skip processing special messages that start with __OUTPUT__
          if (!event.data.startsWith('__OUTPUT__')) {
            xtermRef.current.write(event.data);
          }
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
      console.log('WebSocket onclose event fired, wasClean:', event.wasClean);

      // Mark this as unexpected disconnection if it wasn't a manual close
      const wasUnexpected = !event.wasClean && !connectionStateRef.current.isWebSocketClosing;
      unexpectedDisconnectRef.current = wasUnexpected;
      connectionStateRef.current.isWebSocketClosing = false;

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

            // Use our reference for reconnect timer
            connectionStateRef.current.reconnectTimer = setTimeout(() => {
              connectionStateRef.current.reconnectTimer = null;
              setReconnectTrigger(prev => prev + 1);
            }, backoffDelay);
          } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
            xtermRef.current.writeln('\r\n\x1b[1;31mMaximum reconnection attempts reached. Please try manual reconnect.\x1b[0m');
          }
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error occurred:', error);
      // Don't set disconnected here, let onclose handle it
      if (xtermRef.current) {
        xtermRef.current.writeln('\r\n\x1b[1;31mWebSocket error occurred.\x1b[0m');
      }
    };
  };

  // Connect to WebSocket when parameters change
  useEffect(() => {
    // Use RAF to ensure DOM updates complete before connection
    const frameId = requestAnimationFrame(() => {
      connectWebSocket();
    });

    // Cleanup function - only close on unmount
    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [webSocketUrl, isConnected, terminalReady, reconnectTrigger]);

  // Separate cleanup effect that only runs on unmount
  useEffect(() => {
    // Return cleanup function for component unmount
    return () => {
      console.log('Terminal component unmounting, cleaning up resources');

      // Clean up any pending reconnect timers
      if (connectionStateRef.current.reconnectTimer) {
        clearTimeout(connectionStateRef.current.reconnectTimer);
        connectionStateRef.current.reconnectTimer = null;
      }

      // Close WebSocket connection
      if (wsRef.current) {
        connectionStateRef.current.isWebSocketClosing = true;
        unexpectedDisconnectRef.current = false; // Mark as expected disconnection
        try {
          wsRef.current.close();
        } catch (err) {
          console.error('Error closing WebSocket during unmount:', err);
        }
        wsRef.current = null;
      }

      // Clear any active timeouts
      activeCommandsRef.current.forEach(cmd => {
        if (cmd.updateTimeout) {
          clearTimeout(cmd.updateTimeout);
        }
      });
      activeCommandsRef.current.clear();
    };
  }, []);

  const disconnectTerminal = () => {
    unexpectedDisconnectRef.current = false; // Mark as expected disconnection
    connectionStateRef.current.isWebSocketClosing = true;
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

  // Track whether any commands are in progress
  const hasActiveCommands = activeCommandsRef.current.size > 0;

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
            {hasActiveCommands && (
              <span className="text-yellow-500 ml-2 flex items-center">
                <AlertCircle size={14} className="mr-1 animate-pulse" />
                (Command running)
              </span>
            )}
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

// Wrap the component with React.memo and provide a custom comparison function
export default React.memo(XTerminal, (prevProps, nextProps) => {
  // Only re-render if these specific props change
  return (
    prevProps.webSocketUrl === nextProps.webSocketUrl &&
    prevProps.isConnected === nextProps.isConnected &&
    prevProps.onDisconnect === nextProps.onDisconnect
    // Intentionally not comparing registerExecuteCommand and onTerminalOutput functions
    // as they might be recreated on parent renders
  );
});