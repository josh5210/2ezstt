// Incoming WebSocket events from ezstt server
export interface PartialEvent {
  type: 'partial';
  utterance_id: string;
  text: string;
  revision: number;
  source: string;
  timestamp: number;
}

export interface FinalEvent {
  type: 'final';
  utterance_id: string;
  text: string;
  source: string;
  confidence: number;
  timestamp: number;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
  code?: string;
}

export interface MetricsEvent {
  type: 'metrics';
  [key: string]: any;
}

// Outgoing WebSocket messages to ezstt server
export interface SessionStartMessage {
  type: 'session.start';
  session_id: string;
  speaker_id: string;
  sample_rate: number;
  format: 'pcm_s16le';
  transport: 'binary';
  meta?: {
    app: string;
    [key: string]: any;
  };
}

export interface SessionEndMessage {
  type: 'session.end';
}

export type IncomingEvent = PartialEvent | FinalEvent | ErrorEvent | MetricsEvent;
export type OutgoingMessage = SessionStartMessage | SessionEndMessage;