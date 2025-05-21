export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'thinking';
  content: string;
  isMarkdown?: boolean;
}

export interface DebugEvent {
  type: 'thinking' | 'tool_call' | 'tool_result' | 'token';
  timestamp: Date;
  content: string;
  isContextAware?: boolean;
}

export interface Model {
  model: string;
  displayName?: string;
  reasonerPrompt?: string;
  responderPrompt?: string;
}