import React, { useState, useRef, useEffect, ComponentPropsWithoutRef } from 'react';
import { ArrowUp, Trash2, Terminal, Settings } from 'lucide-react';
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
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  models,
  chatHistory,
  onSendMessage,
  onClearChat,
  onModelSelect,
  onModelDelete,
  onModelPull,
  selectedModel,
  onToggleDebug,
  showDebugPanel,
  isStreaming
}) => {
  const [message, setMessage] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [pullModelName, setPullModelName] = useState('');
  const [isPulling, setIsPulling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [pullStatus, setPullStatus] = useState<{ type: 'success' | 'error' | null; message: string }>({
    type: null,
    message: ''
  });
  const [isModalOpen, setIsModalOpen] = useState(false);

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

  const handlePullModel = async () => {
    if (!pullModelName.trim()) return;

    setIsPulling(true);
    setPullStatus({ type: null, message: '' });

    try {
      await onModelPull(pullModelName);
      setPullStatus({
        type: 'success',
        message: `Model ${pullModelName} pulled successfully`
      });
      setPullModelName(''); // Clear the input after successful pull
    } catch (error) {
      setPullStatus({
        type: 'error',
        message: `Failed to pull model: ${error instanceof Error ? error.message : String(error)}`
      });
    } finally {
      setIsPulling(false);
    }
  };

  const handleDeleteModel = async () => {
    if (!selectedModel) return;

    setIsDeleting(true);

    try {
      await onModelDelete(selectedModel);
    } finally {
      setIsDeleting(false);
    }
  };

  // Auto-hide success message after 5 seconds
  useEffect(() => {
    if (pullStatus.type === 'success') {
      const timer = setTimeout(() => {
        setPullStatus({ type: null, message: '' });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [pullStatus]);

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
      <SyntaxHighlighter style={vscDarkPlus} language={match[1]}>
        {codeString}
      </SyntaxHighlighter>
    );
  };

  return (
    <div className="flex flex-col max-h-[100vh] bg-black rounded-lg overflow-hidden">
      <div className="p-4 bg-black flex items-center justify-between border border-cyan rounded-lg">

        <button
          onClick={onToggleDebug}
          className={`bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg text-sm flex items-center ${
            showDebugPanel ? 'bg-green-600' : ''
          }`}
        >
          <Terminal className="mr-2" size={16} />
          {showDebugPanel ? 'Hide Debug' : 'Show Debug'}
        </button>

        <button
          onClick={() => setIsModalOpen(true)}
          className="bg-cyan hover:bg-bgCyan text-Black px-3 py-2 rounded-lg text-sm flex items-center"
        >
          <Settings className="mr-2" size={16} />
          Models Control
        </button>
      </div>

      {/* Modal for Utilities */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-96">
            <h2 className="text-lg font-bold text-white mb-4">Models Control</h2>

            {/* Pull Model Section */}
            <div className="flex flex-col space-y-2 mb-4">
              <div className="flex space-x-2 items-center">
                <input
                  type="text"
                  value={pullModelName}
                  onChange={(e) => setPullModelName(e.target.value)}
                  placeholder="Model name to pull (e.g. mistral)"
                  className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-2 text-sm"
                  disabled={isPulling}
                />
                <button
                  onClick={handlePullModel}
                  className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg text-sm flex items-center"
                  disabled={isPulling || !pullModelName.trim()}
                >
                  {isPulling ? 'Pulling...' : 'Pull Model'}
                </button>
              </div>

              {pullStatus.type && (
                <div
                  className={`text-sm px-3 py-2 rounded-md ${
                    pullStatus.type === 'success'
                      ? 'bg-green-800 text-green-100'
                      : 'bg-red-800 text-red-100'
                  }`}
                >
                  {pullStatus.message}
                </div>
              )}
            </div>

            <div className="flex items-center">
              <select
                value={selectedModel}
                onChange={(e) => onModelSelect(e.target.value)}
                className="bg-gray-700 text-white rounded-lg px-4 py-2 flex-1 max-w-xs"
              >
                {models.length > 0 ? (
                  models.map((model) => (
                    <option key={model.model} value={model.model}>
                      {model.model}
                    </option>
                  ))
                ) : (
                  <option value="">Loading models...</option>
                )}
              </select>
              <button
                onClick={handleDeleteModel}
                className="bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-2 rounded-lg text-sm ml-2 flex items-center"
                disabled={!selectedModel || isDeleting}
              >
                {isDeleting ? 'Deleting...' : 'Delete Model'}
              </button>
            </div>

            {/* Close Modal Button */}
            <button
              onClick={() => setIsModalOpen(false)}
              className="mt-4 bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Chat heading */}
      <h1 className="text-2xl font-bold text-white text-left mt-4">Chat Interface</h1>

      {/* Chat messages container */}
      <div className="bg-gray-900 flex-1 overflow-y-auto p-4 space-y-4 chat-container mt-5 rounded-lg border border-cyan">
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
        {/* Message input form */}
        <form onSubmit={handleSubmit} className="p-4">
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
          </div>
        </form>
      </div>
    </div>
  );
};