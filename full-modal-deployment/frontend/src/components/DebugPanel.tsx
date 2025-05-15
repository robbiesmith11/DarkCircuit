import React, { useState, useRef, useEffect } from 'react';
import { Trash2 } from 'lucide-react';
import { DebugEvent } from '../types';

interface DebugPanelProps {
  events: DebugEvent[];
  onClear: () => void;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ events, onClear }) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterType, setFilterType] = useState<string | null>(null);

  // Handle auto-scrolling
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll]);

  // Detect manual scrolling to disable auto-scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (container) {
        const { scrollTop, scrollHeight, clientHeight } = container;
        // If user has scrolled up more than 100px from bottom, disable auto-scroll
        setAutoScroll(scrollHeight - scrollTop - clientHeight < 100);
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Format timestamp
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  };

  // Filter events based on selected type
  const filteredEvents = filterType
    ? events.filter(event => event.type === filterType)
    : events;

  return (
    <div className="flex flex-col h-full bg-gray-900">
      <div className="p-2 bg-gray-800 border-b border-gray-700 flex justify-between items-center">
        <h2 className="text-white font-semibold text-sm">Debug Panel</h2>
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <button
              onClick={() => setFilterType(null)}
              className={`px-2 py-1 rounded text-xs ${filterType === null ? 'bg-purple-600' : 'bg-gray-600'}`}
            >
              All
            </button>
            <button
              onClick={() => setFilterType('thinking')}
              className={`px-2 py-1 rounded text-xs ${filterType === 'thinking' ? 'bg-purple-600' : 'bg-gray-600'}`}
            >
              Thinking
            </button>
            <button
              onClick={() => setFilterType('tool_call')}
              className={`px-2 py-1 rounded text-xs ${filterType === 'tool_call' ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              Tool Calls
            </button>
            <button
              onClick={() => setFilterType('tool_result')}
              className={`px-2 py-1 rounded text-xs ${filterType === 'tool_result' ? 'bg-green-600' : 'bg-gray-600'}`}
            >
              Results
            </button>
          </div>
          <button
            onClick={() => setAutoScroll(true)}
            className={`px-2 py-1 rounded text-xs ${autoScroll ? 'bg-blue-600' : 'bg-gray-600'}`}
          >
            Auto-scroll
          </button>
          <button
            onClick={onClear}
            className="bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded-lg flex items-center text-xs"
          >
            <Trash2 className="mr-1" size={12}/>
            Clear
          </button>
        </div>
      </div>

      <div
          ref={containerRef}
          className="flex-grow min-h-0 overflow-y-auto p-2 text-sm font-mono"
      >
        {filteredEvents.length === 0 ? (
            <div className="text-gray-500 italic text-center mt-4">
              {events.length === 0
                ? "No debug events yet. Start a conversation to see what's happening behind the scenes."
                : "No events match the current filter."}
            </div>
        ) : (
            <div className="space-y-1">
              {filteredEvents.map((event, index) => (
                  <div
                      key={index}
                      className={`p-2 rounded border-l-4 ${
                          event.type === 'thinking'
                              ? 'bg-purple-900/30 border-purple-500'
                              : event.type === 'tool_call'
                                  ? 'bg-blue-900/30 border-blue-500'
                                  : event.type === 'tool_result'
                                      ? 'bg-green-900/30 border-green-500'
                                      : 'bg-gray-800 border-gray-500'
                      }`}
                  >
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span className="font-semibold">
                        {event.type === 'thinking'
                            ? '🧠 Thinking'
                            : event.type === 'tool_call'
                                ? '🔧 Tool Call'
                                : event.type === 'tool_result'
                                    ? '📊 Tool Result'
                                    : '💬 Token'}
                      </span>
                      <span>{formatTime(event.timestamp)}</span>
                    </div>
                    <div className="text-white whitespace-pre-wrap">
                      {event.content}
                    </div>
                  </div>
              ))}
            </div>
        )}
        <div ref={bottomRef}/>
      </div>
    </div>
  );
};