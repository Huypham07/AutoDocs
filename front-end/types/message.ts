export interface ChatMessage {
  role: "user" | "assistant";
  message: string;
  timestamp: Date;
  sources?: string[]; // Optional, for assistant messages that include sources
}

export interface ChatResponse {
  message: string;
  sources?: string[];
}
