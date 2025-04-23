import { useEffect, useState, useRef, useCallback } from 'react';
import { ChatInterface } from './ChatInterface';
import { ChatMessage, Model, DebugEvent } from '../types';
import { toast } from 'react-toastify';
import { DebugPanel } from './DebugPanel';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL || '';

// Add a new onSshToolCall prop to receive execute command function
interface ChatContainerProps {
  onSshToolCall?: (command: string, commandId?: number) => Promise<string>;
  selectedChallenge?: string;  // Add selected challenge prop
  targetIp?: string;          // Add target IP prop
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  onSshToolCall,
  selectedChallenge = '',
  targetIp = ''
}) => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [debugEvents, setDebugEvents] = useState<DebugEvent[]>([]);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentAssistantIndexRef = useRef<number>(-1);
  // Add a flag to track if a response is currently streaming
  const [isStreaming, setIsStreaming] = useState(false);
  // Add a flag to track if it's the first message in the conversation
  const [isFirstMessage, setIsFirstMessage] = useState(true);
  // Add a map to store pending terminal commands
  const pendingTerminalCommandsRef = useRef<Map<number, {
    command: string;
    pendingPromise: Promise<string>;
  }>>(new Map());

  // Create a ref to store the current values of selected challenge and target IP
  const selectedChallengeRef = useRef(selectedChallenge);
  const targetIpRef = useRef(targetIp);

  // Update refs when props change
  useEffect(() => {
    selectedChallengeRef.current = selectedChallenge;
  }, [selectedChallenge]);

  useEffect(() => {
    targetIpRef.current = targetIp;
  }, [targetIp]);

  // Add state for system prompts with default values
  const [reasonerPrompt, setReasonerPrompt] = useState<string>("");
  const [responderPrompt, setResponderPrompt] = useState<string>("");

  // Modified to accept prompts as parameters and update all model prompts
  const createModels = useCallback((reasonerPrompt: string, responderPrompt: string) => {
    console.log("Creating models with prompts:", { reasonerPrompt, responderPrompt });

    const predefinedModels: Model[] = [
      {
        model: "gpt-4.1",
        displayName: "GPT-4.1",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "gpt-4o",
        displayName: "GPT-4o",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "gpt-4o-mini",
        displayName: "GPT-4o Mini",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "gpt-3.5-turbo",
        displayName: "GPT-3.5 Turbo",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "o4",
        displayName: "o4",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "o4-mini",
        displayName: "o4-mini",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
      {
        model: "o4-mini-high",
        displayName: "o4-mini-high",
        reasonerPrompt: reasonerPrompt,
        responderPrompt: responderPrompt
      },
    ]

    // Use predefined models if API returns empty
    setModels(predefinedModels);

    // Only set default model if not already set
    if (!selectedModel) {
      setSelectedModel('gpt-4o-mini');
    }
  }, [selectedModel]);

  useEffect(() => {
    const initialize = async () => {
      try {
        // Correct path - starts from the root of your web server
        const response = await fetch('/prompts.json');

        if (response.ok) {
          const prompts = await response.json();

          // Store in state
          setReasonerPrompt(prompts.reasonerPrompt);
          setResponderPrompt(prompts.responderPrompt);

          // Create models with the loaded prompts
          createModels(prompts.reasonerPrompt, prompts.responderPrompt);
        } else {
          console.warn("Couldn't find any default prompts!");
          // Use fallback prompts
          const fallbackReasoner = "Default reasoner prompt if JSON fails to load";
          const fallbackResponder = "Default responder prompt if JSON fails to load";

          setReasonerPrompt(fallbackReasoner);
          setResponderPrompt(fallbackResponder);
          createModels(fallbackReasoner, fallbackResponder);
        }
      } catch (error) {
        console.error("Error loading prompts:", error);
        // Handle error with fallbacks
        const fallbackReasoner = "Default reasoner prompt if JSON fails to load";
        const fallbackResponder = "Default responder prompt if JSON fails to load";

        setReasonerPrompt(fallbackReasoner);
        setResponderPrompt(fallbackResponder);
        createModels(fallbackReasoner, fallbackResponder);
      }
    };

    initialize();

    return () => abortControllerRef.current?.abort();
  }, [createModels]);

  const handleSendMessage = useCallback(async (message: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Reset debug events for new conversation
    setDebugEvents([]);

    // Modify message for first message in conversation - log current values from refs for debugging
    let finalMessage = message;
    if (isFirstMessage) {
      console.log("First message handling - Current values:", {
        selectedChallenge: selectedChallengeRef.current,
        targetIp: targetIpRef.current
      });

      // Explicitly check if either value has content (not just if the ref exists)
      const hasChallenge = selectedChallengeRef.current && selectedChallengeRef.current.trim() !== '';
      const hasTargetIp = targetIpRef.current && targetIpRef.current.trim() !== '';

      if (hasChallenge || hasTargetIp) {
        let suffix = '\n\n'; // Start with line breaks after the original message

        if (hasChallenge) {
          suffix += `HackTheBox challenge: ${selectedChallengeRef.current}\n`;
        }

        if (hasTargetIp) {
          suffix += `Target IP: ${targetIpRef.current}\n`;
        }

        if (suffix !== '\n\n') {
          finalMessage = finalMessage + suffix;
          // Log the modified message to debug
          setDebugEvents(prev => [...prev, {
            type: 'thinking',
            timestamp: new Date(),
            content: `Modified first message with target info: ${finalMessage}`
          }]);
        }
      }

      // No longer the first message after sending
      setIsFirstMessage(false);
    }

    // Prepend chat history to give context to the agent
    // but only if we have previous messages
    if (chatHistory.length > 0) {
      // Create a formatted conversation history string
      let conversationContext = "Previous conversation:\n";

      // Limit to the last 10 messages to avoid context length issues
      const recentMessages = chatHistory.slice(-10);

      recentMessages.forEach(msg => {
        if (msg.role === 'user') {
          conversationContext += `User: ${msg.content}\n`;
        } else if (msg.role === 'assistant') {
          conversationContext += `Assistant: ${msg.content}\n`;
        }
      });

      // Add a separator to distinguish between history and new query
      conversationContext += "\n--- New message ---\n";

      // Combine history with the new message
      finalMessage = conversationContext + finalMessage;

      // Log the context addition to debug panel
      setDebugEvents(prev => [...prev, {
        type: 'thinking',
        timestamp: new Date(),
        content: `Added conversation context to message (${recentMessages.length} previous messages)`
      }]);
    }

    const userMessage: ChatMessage = { role: 'user', content: message }; // Original message for display
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
      // Define the request body with a proper type
      interface ChatRequestBody {
        model: string;
        messages: { role: string; content: string }[];
        reasoner_prompt?: string;
        responder_prompt?: string;
      }

      // Find the current model to get its prompts
      const currentModel = models.find(m => m.model === selectedModel);

      const requestBody: ChatRequestBody = {
        model: selectedModel,
        messages: [{ role: 'user', content: finalMessage }], // Use the final message with context
        reasoner_prompt: currentModel?.reasonerPrompt || reasonerPrompt,
        responder_prompt: currentModel?.responderPrompt || responderPrompt
      };

      console.log("Sending request with prompts:", {
        model: selectedModel,
        reasoner_prompt: requestBody.reasoner_prompt?.substring(0, 50) + "...",
        responder_prompt: requestBody.responder_prompt?.substring(0, 50) + "..."
      });

      const res = await fetch(`${BACKEND_API}/api/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
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
      // Check if this is an AbortError, which means we manually cancelled it
      if (error instanceof DOMException && error.name === 'AbortError') {
        console.log('Request was aborted');
        // We're cleaning up in handleClearChat, so no need to update chat history here
        return;
      }

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
  }, [BACKEND_API, chatHistory.length, isFirstMessage, models, onSshToolCall, reasonerPrompt, responderPrompt, selectedModel]);

  // Handle clearing the chat and resetting the first message flag
  const handleClearChat = useCallback(() => {
    // Abort any ongoing requests first
    if (abortControllerRef.current) {
      console.log('Aborting ongoing request...');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Clear all pending terminal commands
    const pendingCommands = pendingTerminalCommandsRef.current;
    if (pendingCommands.size > 0) {
      console.log(`Clearing ${pendingCommands.size} pending terminal commands`);
      pendingCommands.clear();
    }

    // Reset streaming flag
    setIsStreaming(false);

    // Clear the chat history and debug events
    setChatHistory([]);
    setDebugEvents([]);
    currentAssistantIndexRef.current = -1; // Reset assistant index on chat clear
    setIsFirstMessage(true); // Reset first message flag so the next message will include challenge/IP info

    toast.info("Chat cleared and all operations stopped");
  }, []);

  // Handle prompt updates - FIXED to update the model prompts too
  const handleUpdateSystemPrompts = useCallback((newReasonerPrompt: string, newResponderPrompt: string) => {
    console.log("System prompts updated:", {
      reasoner: newReasonerPrompt.substring(0, 50) + "...",
      responder: newResponderPrompt.substring(0, 50) + "..."
    });

    // Update the base prompts
    setReasonerPrompt(newReasonerPrompt);
    setResponderPrompt(newResponderPrompt);

    // Update all models with new prompts
    const updatedModels = models.map(model => ({
      ...model,
      reasonerPrompt: newReasonerPrompt,
      responderPrompt: newResponderPrompt
    }));

    setModels(updatedModels);

    toast.success("System prompts updated successfully");
  }, [models]);

  // Use memoized toggle debug function
  const toggleDebugPanel = useCallback(() => {
    setShowDebugPanel(prev => !prev);
  }, []);

  // Use memoized function for model selection
  const handleModelSelect = useCallback((model: string) => {
    console.log("Selected model:", model);
    setSelectedModel(model);
  }, []);

  // Use memoized function for debug events clearing
  const handleClearDebugEvents = useCallback(() => {
    setDebugEvents([]);
  }, []);

  return (
    <div className="h-full bg-black flex flex-col max-h-screen">
      {/* Main chat interface - now has a flex-grow but with min-height */}
      <div className="flex-grow min-h-0 overflow-hidden">
        <ChatInterface
          models={models}
          chatHistory={chatHistory}
          onSendMessage={handleSendMessage}
          onClearChat={handleClearChat}
          onModelSelect={handleModelSelect}
          selectedModel={selectedModel}
          onToggleDebug={toggleDebugPanel}
          showDebugPanel={showDebugPanel}
          isStreaming={isStreaming}
          onUpdateSystemPrompts={handleUpdateSystemPrompts}
        />
      </div>

      {/* Debug panel - fixed height with scrolling */}
      {showDebugPanel && (
        <div className="h-64 border-t border-gray-700 overflow-hidden flex-shrink-0">
          <DebugPanel
            events={debugEvents}
            onClear={handleClearDebugEvents}
          />
        </div>
      )}
    </div>
  );
};