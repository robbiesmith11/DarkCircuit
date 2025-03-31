import { useEffect, useState, useRef } from 'react';
import { ChatInterface } from './ChatInterface';
import { ChatMessage, Model } from '../types';
import { toast } from 'react-toastify';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL || '';

export const ChatContainer = () => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
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

    // Add the user's message to the chat history
    const userMessage: ChatMessage = { role: 'user', content: message };
    setChatHistory((prev) => [...prev, userMessage]);

    // Create a placeholder for the assistant's response
    setChatHistory((prev) => [...prev, { role: 'assistant', content: '' }]);

    // Create a new AbortController for this request
    abortControllerRef.current = new AbortController();

    // Make a POST request to /api/chat/completions with a messages array
    fetch(`${BACKEND_API}/api/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        messages: [{ role: 'user', content: message }]
      }),
      signal: abortControllerRef.current.signal
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        if (!data.success) {
          throw new Error(data.error || 'Agent returned an error.');
        }

        // data.messages is the array of { role, content }
        const agentMessages = data.messages || [];
        console.log("agentMessages array:", agentMessages);
        // Overwrite the placeholder assistant message with the agent's response
        setChatHistory((prev) => {
          const newHistory = [...prev];
          const lastIndex = newHistory.findIndex(
            (msg, i) => msg.role === 'assistant' && i === newHistory.length - 1
          );
          if (lastIndex !== -1 && agentMessages.length > 0) {
            newHistory[lastIndex] = {
              ...newHistory[lastIndex],
              content: agentMessages.map((m: ChatMessage) => m.content).join('\n')
            };
          }
          return newHistory;
        });
      })
      .catch((error) => {
        console.error('Error with /api/chat/completions call:', error);
        setChatHistory((prev) => {
          const newHistory = [...prev];
          const lastIndex = newHistory.findIndex(
            (msg, i) => msg.role === 'assistant' && i === newHistory.length - 1
          );
          if (lastIndex !== -1) {
            newHistory[lastIndex] = {
              ...newHistory[lastIndex],
              content: `Error: ${error.message}`
            };
          }
          return newHistory;
        });
      });
  };

  // Handle model deletion with proper error handling and UI feedback
  const handleModelDelete = async (model: string): Promise<void> => {
    if (!model) return;
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
    <div className="h-full">
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
