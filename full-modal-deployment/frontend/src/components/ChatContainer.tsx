import { useEffect, useState, useRef } from 'react';
import { ChatInterface } from './ChatInterface';
import { ChatMessage, Model } from '../types';
import { toast } from 'react-toastify';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL || '';

interface ToolCallEvent {
  tool_call: {
    name: string | { name: string };
    input: string | Record<string, unknown>;
  };
}

interface ToolResultEvent {
  tool_result: {
    output: {
      result?: string;
      content?: string;
      [key: string]: unknown;
    } | string;
  };
}

type StreamedEvent = ToolCallEvent | ToolResultEvent;

export const ChatContainer = () => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const abortControllerRef = useRef<AbortController | null>(null);

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

  const formatToolMessage = (event: StreamedEvent): ChatMessage => {
    if ('tool_call' in event) {
      const tool = event.tool_call;
      const name = typeof tool.name === 'string' ? tool.name : tool.name.name;
      const input = typeof tool.input === 'string' ? tool.input : JSON.stringify(tool.input, null, 2);
      return {
        role: 'system',
        content: `🛠️ **Tool called**: \`${name}\`\n📥 **Input**: \`${input}\``
      };
    }

    if ('tool_result' in event) {
      const raw = event.tool_result.output;
      const output =
        typeof raw === 'string'
          ? raw
          : raw?.content || raw?.result || JSON.stringify(raw, null, 2);
      return {
        role: 'system',
        content: `✅ **Tool result**:\n\`\`\`\n${output}\n\`\`\``
      };
    }

    return { role: 'system', content: '⚠️ Unknown tool event' };
  };

  const handleSendMessage = async (message: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const userMessage: ChatMessage = { role: 'user', content: message };
    setChatHistory((prev) => [...prev, userMessage, { role: 'assistant', content: '' }]);

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
      let assistantContent = '';

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

            if (parsed.tool_call || parsed.tool_result) {
              const toolMessage = formatToolMessage(parsed);
              setChatHistory((prev) => [...prev, toolMessage]);
            }

            if (parsed.choices && parsed.choices[0]?.delta?.content !== undefined) {
              const token = parsed.choices[0].delta.content;
              assistantContent += token;
              setChatHistory((prev) => {
                const updated = [...prev];
                const lastIndex = updated.findIndex(
                  (msg, i) => msg.role === 'assistant' && i === updated.length - 1
                );
                if (lastIndex !== -1) {
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: assistantContent
                  };
                }
                return updated;
              });
            } else if (parsed.type === 'token' && typeof parsed.value === 'string') {
              const token = parsed.value;
              assistantContent += token;
              setChatHistory((prev) => {
                const updated = [...prev];
                const lastIndex = updated.findIndex(
                  (msg, i) => msg.role === 'assistant' && i === updated.length - 1
                );
                if (lastIndex !== -1) {
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: assistantContent
                  };
                }
                return updated;
              });
            }

          } catch (err) {
            console.warn('Parse error:', msg, err);
          }
        }
      }

      // ✅ Force flush at the end
      if (assistantContent) {
        setChatHistory((prev) => {
          const updated = [...prev];
          const lastIndex = updated.findIndex(
            (msg, i) => msg.role === 'assistant' && i === updated.length - 1
          );
          if (lastIndex !== -1) {
            updated[lastIndex] = {
              ...updated[lastIndex],
              content: assistantContent
            };
          }
          return updated;
        });
      }

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error('Streaming error:', errorMsg);
      setChatHistory((prev) => {
        const updated = [...prev];
        const lastIndex = updated.findIndex((msg, i) => msg.role === 'assistant' && i === updated.length - 1);
        if (lastIndex !== -1) {
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: `Error: ${errorMsg}`
          };
        }
        return updated;
      });
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
