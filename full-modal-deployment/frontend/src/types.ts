export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'thinking';
  content: string;
}

export interface DebugEvent {
  type: 'thinking' | 'tool_call' | 'tool_result' | 'token';
  timestamp: Date;
  content: string;
}

export interface Model {
  model: string;
}