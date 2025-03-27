import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Download } from 'lucide-react';
import { ChatMessage, Model } from '../types';

interface ChatInterfaceProps {
  models: Model[];
  chatHistory: ChatMessage[];
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
  onModelSelect: (model: string) => void;
  onModelDelete: (model: string) => void;
  onModelPull: (model: string) => void;
  selectedModel: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  models,
  chatHistory,
  onSendMessage,
  onClearChat,
  onModelSelect,
  onModelDelete,
  onModelPull,
  selectedModel
}) => {
  const [message, setMessage] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [pullModelName, setPullModelName] = useState('');
  const [isPulling, setIsPulling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [pullStatus, setPullStatus] = useState<{type: 'success' | 'error' | null, message: string}>({
    type: null,
    message: ''
  });

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
    setPullStatus({type: null, message: ''});

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
        setPullStatus({type: null, message: ''});
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [pullStatus]);

  return (
      <div className="flex flex-col max-h-[80vh] bg-gray-900 rounded-lg overflow-hidden">
        <div className="p-4 bg-gray-800 border-b border-gray-700">
          <div className="flex flex-col space-y-2">
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
                {isPulling ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Pulling...
                  </span>
                ) : (
                  <>
                    <Download className="mr-2" size={16} />
                    Pull Model
                  </>
                )}
              </button>
            </div>

            {pullStatus.type && (
              <div className={`text-sm px-3 py-2 rounded-md ${pullStatus.type === 'success' ? 'bg-green-800 text-green-100' : 'bg-red-800 text-red-100'}`}>
                {pullStatus.message}
              </div>
            )}
          </div>
        </div>

        <div className="p-4 bg-gray-800 flex items-center justify-between border-b border-gray-700">
          <div className="flex items-center flex-1">
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
              {isDeleting ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Deleting...
                </span>
              ) : (
                <>
                  <Trash2 className="mr-2" size={16} />
                  Delete Model
                </>
              )}
            </button>
          </div>
          <button
              onClick={onClearChat}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center ml-2"
          >
            <Trash2 className="mr-2" size={18}/>
            Clear Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4 chat-container">
          {chatHistory.length > 0 ? (
              chatHistory.map((msg, index) => (
                  <div
                      key={index}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                            msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
                        }`}
                    >
                      <p className="text-sm font-semibold mb-1">
                        {msg.role === 'user' ? 'You' : 'Assistant'}
                      </p>
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
              ))
          ) : (
              <div className="flex justify-center items-center h-full text-gray-500">
                No messages yet. Start a conversation!
              </div>
          )}
          <div ref={chatEndRef}/>
        </div>
        <form onSubmit={handleSubmit} className="p-4 bg-gray-800">
          <div className="flex space-x-2">
            <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask about security tools, techniques or concepts..."
                className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2"
            />
            <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center"
                disabled={!message.trim()}
            >
              <Send className="mr-2" size={18}/>
              Send
            </button>
          </div>
        </form>
      </div>
  );
};