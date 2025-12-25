import WebSocket from 'ws';

export type OutEvent = { type: 'partial'|'final'|'error'|'metrics'; [k: string]: any };

export class WSSession {
  private ws?: WebSocket;
  private queue: Buffer[] = [];
  private maxQueue = 200; // ~4s
  private open = false;
  private flushScheduled = false;
  private endRequested = false;
  private closing = false;
  private closeTimer?: NodeJS.Timeout;
  private readonly endGraceMs = 4000;
  constructor(private url: string, private meta: { sessionId: string; speakerId: string; sampleRate?: number }) {}

  connect(onEvent: (ev: OutEvent)=>void): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url, { perMessageDeflate: false });
      this.ws.on('open', () => {
        this.open = true;
        const start = { type: 'session.start', session_id: this.meta.sessionId, speaker_id: this.meta.speakerId, sample_rate: this.meta.sampleRate ?? 16000, format: 'pcm_s16le', transport: 'binary', meta: { app: 'ezstt-client' } };
        this.ws!.send(JSON.stringify(start));
        resolve();
      });
      this.ws.on('message', (data) => {
        try {
          const payload = JSON.parse(data.toString('utf8')) as OutEvent;
          onEvent(payload);
          if (this.endRequested && payload.type === 'final') {
            this.closeSoon(0);
          }
        } catch {}
      });
      this.ws.on('close', () => {
        this.open = false;
        this.queue = [];
        this.flushScheduled = false;
        this.endRequested = false;
        this.closing = true;
        if (this.closeTimer) {
          clearTimeout(this.closeTimer);
          this.closeTimer = undefined;
        }
      });
      this.ws.on('error', (e) => { reject(e); });
    });
  }

  writeFrame(frame: Buffer) {
    if (!this.ws || !this.open) return;
    // backpressure: drop oldest when too many
    if (this.queue.length >= this.maxQueue) this.queue.shift();
    this.queue.push(frame);
    this.scheduleFlush();
  }

  private scheduleFlush() {
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    setTimeout(() => {
      this.flushScheduled = false;
      this.flush();
    }, 0);
  }

  private flushing = false;
  private flush() {
    if (this.flushing || !this.ws || !this.open) return;
    this.flushing = true;
    while (this.queue.length && this.open) {
      const buf = this.queue.shift()!;
      // Send as raw binary. Server slices into frames already aligned to 20ms
      this.ws!.send(buf, { binary: true });
    }
    this.flushing = false;
  }

  end() {
    if (!this.ws) return;
    if (this.endRequested) return;
    this.endRequested = true;
    const sendEnd = () => {
      try { this.ws!.send(JSON.stringify({ type: 'session.end' })); } catch {}
    };
    if (this.ws.readyState === WebSocket.OPEN) {
      sendEnd();
    } else {
      this.ws.once('open', sendEnd);
    }
    this.closeSoon(this.endGraceMs);
  }

  private closeSoon(delayMs: number) {
    if (!this.ws || this.closing) return;
    if (delayMs <= 0) {
      this.closing = true;
      try { this.ws.close(); } catch {}
      return;
    }
    if (this.closeTimer) return;
    this.closeTimer = setTimeout(() => {
      this.closing = true;
      try { this.ws?.close(); } catch {}
      this.closeTimer = undefined;
    }, delayMs);
  }
}
