import { useEffect, useState, useRef } from 'react';
import { ChatInterface } from './ChatInterface';
import { ChatMessage, Model, DebugEvent } from '../types';
import { toast } from 'react-toastify';
import { DebugPanel } from './DebugPanel';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL || '';

// Add a new onSshToolCall prop to receive execute command function
interface ChatContainerProps {
  onSshToolCall?: (command: string, commandId?: number) => Promise<string>;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({ onSshToolCall }) => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [debugEvents, setDebugEvents] = useState<DebugEvent[]>([]);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentAssistantIndexRef = useRef<number>(-1);
  // Add a flag to track if a response is currently streaming
  const [isStreaming, setIsStreaming] = useState(false);
  // Add a map to store pending terminal commands
  const pendingTerminalCommandsRef = useRef<Map<number, {
    command: string;
    pendingPromise: Promise<string>;
  }>>(new Map());

  const fetchModels = async () => {
    try {
      const res = await fetch(`${BACKEND_API}/api/models`);
      const data = await res.json();
      if (data.models?.length) {
        setModels(data.models);
        if (!selectedModel || !data.models.some((m: Model) => m.model === selectedModel)) {
          setSelectedModel(data.models[0].model);
        }
      } else {
        setModels([]);
        setSelectedModel('');
      }
    } catch (error) {
      toast.error(`Failed to fetch models: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  useEffect(() => {
    fetchModels();
    return () => abortControllerRef.current?.abort();
  }, []);

  const handleSendMessage = async (message: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Reset debug events for new conversation
    setDebugEvents([]);

    const userMessage: ChatMessage = { role: 'user', content: message };
    // Add user message and create an empty assistant message
    setChatHistory((prev) => [...prev, userMessage, { role: 'assistant', content: '', isMarkdown: false }]);

    // Set streaming flag to true
    setIsStreaming(true);

    // Set the current assistant index to the newly added assistant message
    const newAssistantIndex = chatHistory.length + 1; // +1 for user message we just added
    currentAssistantIndexRef.current = newAssistantIndex;

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    try {
      const res = await fetch(`${BACKEND_API}/api/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          messages: [{ role: 'user', content: message }]
        }),
        signal
      });

      if (!res.ok || !res.body) throw new Error(`HTTP error ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantContentStarted = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const messages = chunk.split('\n\n');

        for (const msg of messages) {
          if (!msg.startsWith('data: ')) continue;
          const data = msg.replace('data: ', '').trim();
          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            console.log("Parsed data:", parsed);

            // Log actual structure of the event to help with debugging
            console.log("Event structure:", Object.keys(parsed));

            // Handle UI terminal commands
            if (parsed.type === 'ui_terminal_command' && onSshToolCall) {
              const commandId = parsed.command_id;
              const command = parsed.command;

              console.log(`🖥️ UI Terminal Command: ${command} (ID: ${commandId})`);

              // Add to debug panel
              setDebugEvents(prev => [...prev, {
                type: 'tool_call',
                timestamp: new Date(),
                content: `Tool: Terminal Command\nCommand ID: ${commandId}\nCommand: ${command}`
              }]);

              // Execute the command in the terminal and get output
              const pendingPromise = onSshToolCall(command, commandId);

              // Store the promise for later use
              pendingTerminalCommandsRef.current.set(commandId, {
                command,
                pendingPromise
              });

              // When the promise resolves, log the output
              pendingPromise.then(output => {
                console.log(`🖥️ Terminal Output for Command ${commandId}: ${output.substring(0, 100)}...`);

                // Add to debug panel
                setDebugEvents(prev => [...prev, {
                  type: 'tool_result',
                  timestamp: new Date(),
                  content: `Command ${commandId} Result:\n${output}`
                }]);

                // Clean up
                pendingTerminalCommandsRef.current.delete(commandId);
              }).catch(error => {
                console.error(`Error executing terminal command ${commandId}:`, error);
                setDebugEvents(prev => [...prev, {
                  type: 'tool_result',
                  timestamp: new Date(),
                  content: `Command ${commandId} Error: ${error.message || String(error)}`
                }]);
                pendingTerminalCommandsRef.current.delete(commandId);
              });
            }

            // Handle standardized tool calls - add to debug panel and check for SSH commands
            else if (parsed.type === 'tool_call') {
              console.log("💥 TOOL CALL DETECTED:", parsed);

              // Extract tool name and input with proper fallbacks
              const toolName = parsed.name || 'unnamed tool';
              const toolDesc = parsed.description || 'unnamed';
              const toolInput = typeof parsed.input === 'string'
                ? parsed.input
                : JSON.stringify(parsed.input, null, 2);

              // Add to debug panel
              setDebugEvents(prev => [...prev, {
                type: 'tool_call',
                timestamp: new Date(),
                content: `Tool: ${toolName}\nDescription: ${toolDesc}\nInput: ${toolInput}`
              }]);

            }

            // Handle standardized tool results - only add to debug panel
            else if (parsed.type === 'tool_result') {
              console.log("💥 TOOL RESULT DETECTED:", parsed);

              // Extract result content with proper fallbacks
              const resultContent = typeof parsed.output === 'string'
                ? parsed.output
                : JSON.stringify(parsed.output, null, 2);

              // Add to debug panel
              setDebugEvents(prev => [...prev, {
                type: 'tool_result',
                timestamp: new Date(),
                content: resultContent
              }]);
            }

            // Handle thinking events - only add to debug panel
            else if (parsed.type === 'thinking' && typeof parsed.value === 'string') {
              // Only add to debug panel, not to chat history
              setDebugEvents(prev => [...prev, {
                type: 'thinking',
                timestamp: new Date(),
                content: parsed.value
              }]);
            }

            // Handle normal content tokens (both OpenAI format and custom)
            else if (
              (parsed.choices && parsed.choices[0]?.delta?.content !== undefined) ||
              (parsed.type === 'token' && typeof parsed.value === 'string')
            ) {
              const token = parsed.choices
                ? parsed.choices[0].delta.content
                : parsed.value;

              if (typeof token === 'string') {
                assistantContentStarted = true;

                // Add meaningful tokens to assistant's response
                setChatHistory(prev => {
                  const updated = [...prev];
                  // Use the stored index to ensure we're updating the correct message
                  const assistantIndex = currentAssistantIndexRef.current;

                  if (assistantIndex >= 0 && assistantIndex < updated.length) {
                    // Get current content and append the new token
                    const currentContent = updated[assistantIndex].content || '';
                    updated[assistantIndex] = {
                      ...updated[assistantIndex],
                      content: currentContent + token,
                      isMarkdown: false, // Keep as false while streaming
                    };
                  }
                  return updated;
                });
              }
              // First empty tokens with no thinking yet
              else if (!assistantContentStarted) {
                setDebugEvents(prev => [...prev, {
                  type: 'thinking',
                  timestamp: new Date(),
                  content: "Processing request..."
                }]);
                assistantContentStarted = true;
              }
            }

          } catch (err) {
            console.warn('Parse error:', msg, err);
          }
        }
      }

      // Streaming is complete, set markdown flag to true
      setChatHistory(prev => {
        const updated = [...prev];
        const assistantIndex = currentAssistantIndexRef.current;

        if (assistantIndex >= 0 && assistantIndex < updated.length) {
          updated[assistantIndex] = {
            ...updated[assistantIndex],
            isMarkdown: true, // Now mark as markdown to trigger rendering
          };
        }
        return updated;
      });

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error('Streaming error:', errorMsg);
      setChatHistory((prev) => {
        const updated = [...prev];
        const assistantIndex = currentAssistantIndexRef.current;
        if (assistantIndex >= 0 && assistantIndex < updated.length) {
          updated[assistantIndex] = {
            ...updated[assistantIndex],
            content: `Error: ${errorMsg}`,
            isMarkdown: false, // Errors don't need markdown
          };
        }
        return updated;
      });
    } finally {
      // Set streaming flag to false when complete
      setIsStreaming(false);
    }
  };

  const handleModelDelete = async (model: string) => {
    if (!model) return;
    try {
      const res = await fetch(`${BACKEND_API}/api/models/${model}`, { method: 'DELETE' });
      const result = await res.json();
      if (!res.ok || !result.success) throw new Error(result.error || 'Delete failed');
      toast.success(`Model ${model} deleted`);
      await fetchModels();
    } catch (err) {
      toast.error(`Failed to delete model: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleModelPull = async (model: string) => {
    if (!model) throw new Error('No model specified');
    try {
      const res = await fetch(`${BACKEND_API}/api/models/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      const result = await res.json();
      if (!res.ok || !result.success) throw new Error(result.error || 'Pull failed');
      toast.success(`Model ${model} pulled successfully`);
      await fetchModels();
    } catch (err) {
      toast.error(`Failed to pull model: ${err instanceof Error ? err.message : String(err)}`);
      throw err;
    }
  };

  return (
    <div className="h-full bg-black flex flex-col max-h-screen">
      {/* Main chat interface - now has a flex-grow but with min-height */}
      <div className="flex-grow min-h-0 overflow-hidden">
        <ChatInterface
          models={models}
          chatHistory={chatHistory}
          onSendMessage={handleSendMessage}
          onClearChat={() => {
            setChatHistory([]);
            setDebugEvents([]);
            currentAssistantIndexRef.current = -1; // Reset assistant index on chat clear
          }}
          onModelSelect={setSelectedModel}
          onModelDelete={handleModelDelete}
          onModelPull={handleModelPull}
          selectedModel={selectedModel}
          onToggleDebug={() => setShowDebugPanel(!showDebugPanel)}
          showDebugPanel={showDebugPanel}
          isStreaming={isStreaming}
        />
      </div>

      {/* Debug panel - fixed height with scrolling */}
      {showDebugPanel && (
        <div className="h-64 border-t border-gray-700 overflow-hidden flex-shrink-0">
          <DebugPanel
            events={debugEvents}
            onClear={() => setDebugEvents([])}
          />
        </div>
      )}
    </div>
  );
};
