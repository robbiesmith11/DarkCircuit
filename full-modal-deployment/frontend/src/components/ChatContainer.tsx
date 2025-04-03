import { useEffect, useState, useRef } from 'react';
import { ChatInterface } from './ChatInterface';
import { ChatMessage, Model } from '../types';
import { toast } from 'react-toastify';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL || '';

export const ChatContainer = () => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  // Removed unused isLoading state
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch models from the API
  const fetchModels = async () => {
    try {
      const res = await fetch(`${BACKEND_API}/api/models`);
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }

      const data = await res.json();
      console.log('Fetched models:', data);

      if (data.models && data.models.length > 0) {
        setModels(data.models);

        // Only set selected model if it's not already set or if current selection is no longer available
        if (!selectedModel || !data.models.some((model: Model) => model.model === selectedModel)) {
          setSelectedModel(data.models[0].model);
          console.log('Selected Model:', data.models[0].model);
        }
      } else {
        setModels([]);
        setSelectedModel('');
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      toast.error(`Failed to fetch models: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  useEffect(() => {
    console.log('ChatContainer mounted');
    fetchModels();

    // Clean up function
    return () => {
      console.log('ChatContainer unmounted');
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Handle sending messages
  const handleSendMessage = (message: string) => {
    // Abort any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const userMessage: ChatMessage = { role: 'user', content: message };
    setChatHistory((prev) => [...prev, userMessage]);

    // Create messages array for API request
    const messages = [...chatHistory, userMessage].map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    // Create a placeholder for the assistant's response
    setChatHistory((prev) => [...prev, {
      role: 'assistant',
      content: ''
    }]);

    // Create a new AbortController for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // Create the fetch request for SSE
    fetch(`${BACKEND_API}/api/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: selectedModel,
        messages: messages
      }),
      signal
    }).then(response => {
      if (!response.body) {
        throw new Error('ReadableStream not supported in this browser.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';

      function processChunks(): Promise<void> {
        return reader.read().then(({ done, value }) => {
          if (done) {
            console.log('Stream complete');
            return;
          }

          buffer += decoder.decode(value, { stream: true });

          // Process buffer content
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || ''; // Keep the last incomplete chunk in the buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              let data = line.slice(6); // Remove 'data: ' prefix

              if (data === '[DONE]') {
                console.log('Stream finished');
                continue;
              }

              try {
                const parsed = JSON.parse(data);

                // Extract the content from the chunk
                const content = parsed.choices[0].delta.content || '';

                // Update the last assistant message
                setChatHistory(prev => {
                  const newHistory = [...prev];
                  const lastMessageIndex = newHistory.length - 1;

                  if (lastMessageIndex >= 0 && newHistory[lastMessageIndex].role === 'assistant') {
                    newHistory[lastMessageIndex] = {
                      ...newHistory[lastMessageIndex],
                      content: newHistory[lastMessageIndex].content + content
                    };
                  }

                  return newHistory;
                });
              } catch (error) {
                console.error('Error parsing SSE data:', error, data);
              }
            }
          }

          return processChunks();
        }).catch(error => {
          if (error.name === 'AbortError') {
            console.log('Fetch aborted');
          } else {
            console.error('Error processing stream:', error);
          }
        });
      }

      return processChunks();
    }).catch(error => {
      if (error.name === 'AbortError') {
        console.log('Fetch aborted');
      } else {
        console.error('Error with streaming:', error);
        setChatHistory(prev => {
          const newHistory = [...prev];
          const lastMessageIndex = newHistory.length - 1;

          if (lastMessageIndex >= 0 && newHistory[lastMessageIndex].role === 'assistant') {
            newHistory[lastMessageIndex] = {
              ...newHistory[lastMessageIndex],
              content: 'Error: Could not connect to the chat service. Please try again later.'
            };
          }

          return newHistory;
        });
      }
    });
  };

  // Handle model deletion with proper error handling and UI feedback
  const handleModelDelete = async (model: string): Promise<void> => {
    if (!model) return;

    // Starting deletion process
    try {
      const res = await fetch(`${BACKEND_API}/api/models/${model}`, {
        method: 'DELETE'
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || `Failed with status: ${res.status}`);
      }

      const result = await res.json();

      if (result.success) {
        toast.success(`Model ${model} deleted successfully`);
        // Refresh the models list
        await fetchModels();
      } else {
        throw new Error(result.error || 'Unknown error occurred');
      }
    } catch (err) {
      console.error("Failed to delete model:", err);
      toast.error(`Failed to delete model: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Handle model pull with proper error handling and UI feedback
  const handleModelPull = async (model: string): Promise<void> => {
    if (!model) throw new Error('No model specified');

    // Starting pull process
    try {
      const res = await fetch(`${BACKEND_API}/api/models/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || `Failed with status: ${res.status}`);
      }

      const result = await res.json();

      if (result.success) {
        toast.success(`Model ${model} pulled successfully`);
        // Refresh the models list
        await fetchModels();
        return;
      } else {
        throw new Error(result.error || 'Unknown error occurred');
      }
    } catch (err) {
      console.error("Failed to pull model:", err);
      toast.error(`Failed to pull model: ${err instanceof Error ? err.message : String(err)}`);
      throw err;
    }
  };

  return (
    <div className="h-full bg-black">
      <ChatInterface
        models={models}
        chatHistory={chatHistory}
        onSendMessage={handleSendMessage}
        onClearChat={() => setChatHistory([])}
        onModelSelect={setSelectedModel}
        onModelDelete={handleModelDelete}
        onModelPull={handleModelPull}
        selectedModel={selectedModel}
      />
    </div>
  );
};