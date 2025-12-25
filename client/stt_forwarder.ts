// client/stt_forwarder.ts
// Tiny WebSocket forwarder for STT out-events → external consumers
// Usage:
//   import { startForwarder, broadcast as forwardSTT } from "./stt_forwarder";
//   startForwarder();            // once at startup
//   forwardSTT(ev);              // on every out-event you receive from ezstt

import WebSocket, { WebSocketServer } from "ws";

type Json = Record<string, unknown>;
type STTEvent = Json & {
  type?: string;           // "partial" | "final" | ...
  text?: string;
  utterance_id?: string;
  speaker_id?: string;
};

const DEFAULT_PORT = Number(process.env.FORWARD_WS_PORT ?? 9002);
// If set to "1", only forward events where ev.type === "final"
const ONLY_FINALS = process.env.FORWARD_ONLY_FINALS === "1";

// Drop partials if any client is too backed up (bytes that haven’t flushed yet)
const PARTIAL_DROP_THRESHOLD_BYTES = 512 * 1024; // 512 KB

class STTForwarder {
  private wss?: WebSocketServer;
  private clients = new Set<WebSocket>();
  private heartbeatTimer?: ReturnType<typeof setInterval>;

  start(port: number = DEFAULT_PORT) {
    if (this.wss) return; // idempotent
    const host = process.env.FORWARD_WS_HOST ?? "0.0.0.0";
    this.wss = new WebSocketServer({ host, port });
    console.log(`[stt_forwarder] listening on ws://${host}:${port}`);

    this.wss.on("connection", (ws, req) => {
      this.clients.add(ws);
      const ip = req.socket.remoteAddress ?? "unknown";
      console.log(`[stt_forwarder] client connected (${ip}), total=${this.clients.size}`);

      // Optional hello
      this.safeSend(ws, { type: "hello", role: "stt_forwarder", ts: Date.now() });

      ws.on("close", () => {
        this.clients.delete(ws);
        console.log(`[stt_forwarder] client disconnected, total=${this.clients.size}`);
      });

      ws.on("error", (err) => {
        console.warn(`[stt_forwarder] client error: ${(err as Error).message}`);
      });
    });

    // Lightweight heartbeat to keep connections alive
    this.heartbeatTimer = setInterval(() => {
      for (const ws of this.clients) {
        if (ws.readyState === WebSocket.OPEN) {
          try { ws.ping(); } catch { /* ignore */ }
        }
      }
    }, 15000);
  }

  stop() {
    if (!this.wss) return;
    if (this.heartbeatTimer) {
        clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = undefined;
    }
    for (const ws of this.clients) {
      try { ws.close(1001, "shutting down"); } catch { /* ignore */ }
    }
    this.clients.clear();
    this.wss.close();
    this.wss = undefined;
    console.log("[stt_forwarder] stopped");
  }

  broadcast(ev: unknown) {
    // Normalize to object; keep original if already stringified
    let obj: STTEvent | undefined;
    if (typeof ev === "string") {
      try { obj = JSON.parse(ev) as STTEvent; } catch { /* non-JSON strings are ignored */ }
    } else if (ev && typeof ev === "object") {
      obj = ev as STTEvent;
    }

    // If we can’t understand the shape, bail quietly
    if (!obj) return;

    // Optional filtering
    if (ONLY_FINALS && obj.type !== "final") return;

    const payload = JSON.stringify(obj);

    for (const ws of this.clients) {
      if (ws.readyState !== WebSocket.OPEN) continue;

      // Backpressure policy: never drop finals; may drop partials if client is backed up
      const isFinal = obj.type === "final";
      const buffered = ws.bufferedAmount ?? 0;
      if (!isFinal && buffered > PARTIAL_DROP_THRESHOLD_BYTES) {
        // Skip this partial for this client; continue to the next client
        continue;
      }

      try {
        ws.send(payload);
      } catch (err) {
        console.warn(`[stt_forwarder] send error: ${(err as Error).message}`);
      }
    }
  }

  private safeSend(ws: WebSocket, data: Json) {
    if (ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify(data)); } catch { /* ignore */ }
    }
  }
}

// Singleton instance + simple helpers
const _forwarder = new STTForwarder();

export function startForwarder(port?: number) {
  _forwarder.start(port);
}

export function stopForwarder() {
  _forwarder.stop();
}

export function broadcast(ev: unknown) {
  _forwarder.broadcast(ev);
}
