import React, { useState, useRef, useEffect, ComponentPropsWithoutRef } from 'react';
import { ArrowUp, Trash2, Terminal, Settings, ArrowDown } from 'lucide-react';
import { ChatMessage, Model } from '../types';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

// Define more accurate types for ReactMarkdown code components
type ReactMarkdownCodeComponentProps = ComponentPropsWithoutRef<'code'> & {
  inline?: boolean;
  className?: string;
  node?: unknown;
  children: React.ReactNode;
};

interface ChatInterfaceProps {
  models: Model[];
  chatHistory: ChatMessage[];
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
  onModelSelect: (model: string) => void;
  onModelDelete: (model: string) => void;
  onModelPull: (model: string) => void;
  selectedModel: string;
  onToggleDebug: () => void;
  showDebugPanel: boolean;
  isStreaming: boolean;
  onUpdateSystemPrompts?: (reasonerPrompt: string, responderPrompt: string) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  models,
  chatHistory,
  onSendMessage,
  onClearChat,
  onModelSelect,
  selectedModel,
  onToggleDebug,
  showDebugPanel,
  isStreaming,
  onUpdateSystemPrompts
}) => {
  const [message, setMessage] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isScrolledUp, setIsScrolledUp] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Initialize system prompts with empty strings - they will be populated when the modal opens
  const [reasonerPrompt, setReasonerPrompt] = useState<string>('');
  const [responderPrompt, setResponderPrompt] = useState<string>('');

  // We'll use the models from props instead of a local predefined list

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleUpdateSystemPrompts = () => {
    if (onUpdateSystemPrompts) {
      onUpdateSystemPrompts(reasonerPrompt, responderPrompt);
    }

    // Close the modal
    setIsModalOpen(false);
  };

  // Scroll to bottom when the user scrolls up
  useEffect(() => {
    const chatContainer = chatContainerRef.current;

    const handleScroll = () => {
      if (chatContainer) {
        const isAtBottom =
          chatContainer.scrollHeight - chatContainer.scrollTop === chatContainer.clientHeight;
        setIsScrolledUp(!isAtBottom);
      }
    };

    if (chatContainer) {
      chatContainer.addEventListener('scroll', handleScroll);
    }

    return () => {
      if (chatContainer) {
        chatContainer.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Effect to populate reasoner and responder prompts from props when modal opens
  useEffect(() => {
    if (isModalOpen) {
      // Find the currently selected model
      const currentModel = models.find(m => m.model === selectedModel);

      // If we have prompts from the current model, use those
      if (currentModel?.reasonerPrompt) {
        setReasonerPrompt(currentModel.reasonerPrompt);
      }

      if (currentModel?.responderPrompt) {
        setResponderPrompt(currentModel.responderPrompt);
      }
    }
  }, [isModalOpen, models, selectedModel]);

  // Custom code component for ReactMarkdown with proper typing
  const CodeBlock = ({ inline, className, children, ...props }: ReactMarkdownCodeComponentProps) => {
    const match = /language-(\w+)/.exec(className || '');

    if (inline || !match) {
      return (
        <code className="inline-code" {...props}>
          {children}
        </code>
      );
    }

    // Ensure the content is a string
    const codeString = Array.isArray(children)
      ? children.join('')
      : typeof children === 'string'
      ? children
      : String(children); // Fallback

    return (
      <SyntaxHighlighter sfftyle={vscDarkPlus} language={match[1]}>
        {codeString}
      </SyntaxHighlighter>
    );
  };

  return (
    <div className="flex flex-col max-h-[100vh] bg-black rounded-lg overflow-hidden">
      {/* Modified Modal for Models Control */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <h2 className="text-lg font-bold text-white mb-4">AI Assistant Configuration</h2>

            {/* Model Selection */}
            <div className="mb-6">
              <label className="block text-white text-sm font-medium mb-2">Select Language Model</label>
              <select
                value={selectedModel}
                onChange={(e) => onModelSelect(e.target.value)}
                className="bg-gray-700 text-white rounded-lg px-4 py-2 w-full"
              >
                {models.length > 0 ? (
                  models.map((model) => (
                    <option key={model.model} value={model.model}>
                      {model.displayName || model.model}
                    </option>
                  ))
                ) : (
                  <option value="">Loading models...</option>
                )}
              </select>
            </div>

            {/* Reasoner Prompt Configuration */}
            <div className="mb-6">
              <label className="block text-white text-sm font-medium mb-2">
                Reasoning System Prompt
                <span className="text-gray-400 ml-2">(How the agent thinks through problems)</span>
              </label>
              <textarea
                value={reasonerPrompt}
                onChange={(e) => setReasonerPrompt(e.target.value)}
                placeholder="Enter custom system prompt for the reasoning step..."
                className="bg-gray-700 text-white rounded-lg px-4 py-2 w-full h-40 resize-y"
              />
            </div>

            {/* Responder Prompt Configuration */}
            <div className="mb-6">
              <label className="block text-white text-sm font-medium mb-2">
                Response System Prompt
                <span className="text-gray-400 ml-2">(How the agent formats final answers)</span>
              </label>
              <textarea
                value={responderPrompt}
                onChange={(e) => setResponderPrompt(e.target.value)}
                placeholder="Enter custom system prompt for the response step..."
                className="bg-gray-700 text-white rounded-lg px-4 py-2 w-full h-40 resize-y"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-between">
              <button
                onClick={() => setIsModalOpen(false)}
                className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdateSystemPrompts}
                className="bg-cyan hover:bg-bgCyan text-black px-4 py-2 rounded-lg"
              >
                Apply Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Chat heading and Models Control button */}
      <div className="flex items-center justify-between mt-4">
        <h1 className="text-2xl font-bold text-white">Chat Interface</h1>
        <button
          onClick={() => setIsModalOpen(true)}
          className="bg-cyan hover:bg-bgCyan text-Black px-3 py-2 rounded-lg text-sm flex items-center"
        >
          <Settings className="mr-2" size={16} />
          AI Configuration
        </button>
      </div>

      {/* Chat messages container */}
      <div
        ref={chatContainerRef} // Attach the ref to the chat container
        className="bg-gray-900 flex-1 overflow-y-auto p-4 space-y-4 chat-container mt-5 rounded-lg border border-cyan relative">
        {chatHistory.length > 0 ? (
          chatHistory.map((msg, index) => {
            // Only show user and assistant messages in main chat
            if (msg.role !== 'user' && msg.role !== 'assistant') {
              return null;
            }

            return (
              <div
                key={index}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    msg.role === 'user'
                      ? 'bg-bgCyan text-black'
                      : 'bg-gray-700 text-bgCyan'
                  }`}
                >
                  <p className="text-sm font-semibold mb-1">
                    {msg.role === 'user' ? 'You' : '🤖 Assistant'}
                  </p>
                  {msg.role === 'assistant' && msg.isMarkdown ? (
                    // Use ReactMarkdown for completed messages
                    <div className="markdown-content">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // Using type assertion with more specific type instead of 'any'
                          code: CodeBlock as React.ComponentType<React.ClassAttributes<HTMLElement> & React.HTMLAttributes<HTMLElement>>
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    // Use the same whitespace handling as the debug panel
                    <div className="text-white whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          <div className="flex justify-center items-center h-20 text-gray-500 select-none">
            No messages yet. Start a conversation!
          </div>
        )}
        <div ref={chatEndRef} />

        {isScrolledUp && (
          <button
            onClick={scrollToBottom}
            className="sticky bottom-[6rem] left-1/2 transform -translate-x-1/2 bg-cyan text-black w-8 h-16 rounded-full shadow-lg hover:bg-bgCyan flex items-center justify-center"
            aria-label="Scroll to bottom"
            title="Scroll to bottom"
          >
            <ArrowDown size={16} />
          </button>
        )}

        {/* Message input form */}
        <form onSubmit={handleSubmit} className="p-4 sticky bottom-0 rounded-lg bg-chat">
          <div className="flex space-x-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask about security tools, techniques or concepts..."
                className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 pr-12"
                disabled={isStreaming}
              />
              <button
                type="submit"
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-cyan hover:bg-bgCyan hover:text-white text-black w-8 h-8 rounded-full flex items-center justify-center"
                disabled={!message.trim() || isStreaming}
              >
                <ArrowUp size={18} />
              </button>
            </div>
            <button
              type="button"
              onClick={onClearChat}
              className="bg-red-600 hover:bg-white hover:text-red-600 text-white w-10 h-10 rounded-full flex items-center justify-center"
              aria-label="Clear chat"
              title="Clear chat"
            >
              <Trash2 size={18} />
            </button>
            <button
              onClick={onToggleDebug}
              className={`bg-green-600 hover:bg-green-800 text-white px-3 py-2 rounded-lg text-sm flex items-center ${
                showDebugPanel ? 'bg-green-600' : ''
              }`}
            >
              <Terminal className="mr-2" size={16} />
              {showDebugPanel ? 'Hide Debug' : 'Show Debug'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};