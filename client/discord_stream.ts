import 'dotenv/config';
import { Client, Events, GatewayIntentBits, TextChannel } from 'discord.js';
import { joinVoiceChannel, EndBehaviorType, getVoiceConnection, VoiceConnection, createAudioPlayer, createAudioResource, AudioPlayerStatus, StreamType } from '@discordjs/voice';
import prism from 'prism-media';
import ffmpeg from 'ffmpeg-static';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import http from 'node:http';
import { FrameChunker } from './audio/frame_chunker';
import { AudioGate } from './audio/audio_gate';
import { WSSession } from './ws_session';
import type { OutEvent } from './ws_session';
import { sendFinalSimpleMessage } from './finals_channel';
import { randomUUID } from 'node:crypto';
import { broadcast, startForwarder } from './stt_forwarder';

// Discord dependency check
import { generateDependencyReport } from '@discordjs/voice';
console.log(generateDependencyReport());

const TOKEN = process.env.DISCORD_TOKEN!;
const GUILD_ID = process.env.GUILD_ID!;
const VOICE_CHANNEL_ID = process.env.VOICE_CHANNEL_ID!;
const TRANSCRIPT_CHANNEL_ID = process.env.TRANSCRIPT_CHANNEL_ID!;
const FINALS_TRANSCRIPT_CHANNEL_ID = process.env.FINALS_TRANSCRIPT_CHANNEL_ID;
const WS_URL = process.env.EZSTT_WS_URL!;
const PARTIALS_ENABLE = process.env.PARTIALS_ENABLE !== '0';
const TTS_PLAYER_PORT = parseInt(process.env.TTS_PLAYER_PORT || '5056', 10);

const DISCORD_MESSAGE_LIMIT = 2000;
const PARTIAL_TRUNCATION_NOTICE = '... older revisions truncated ...';
const MESSAGE_TRIM_NOTICE = '... message trimmed ...';

type PartialEventPayload = {
  type: 'partial';
  session_id: string;
  speaker_id: string;
  utterance_id: string;
  revision: number;
  text: string;
  t0_ms: number;
  t1_ms: number;
  confidence: number;
  low_confidence?: boolean;
  latency_ms?: number;
  source?: string;
  [key: string]: any;
};

type FinalEventPayload = {
  type: 'final';
  session_id: string;
  speaker_id: string;
  utterance_id: string;
  text: string;
  confidence: number;
  source?: string;
  t0_ms?: number;
  t1_ms?: number;
  frames?: number;
  voiced_ms?: number;
  silence_tail_ms?: number;
  dropped_frames?: number;
  latency_ms?: number;
  [key: string]: any;
};

interface PartialRevision {
  revision: number;
  text: string;
  t0_ms: number;
  t1_ms: number;
  confidence: number;
  lowConfidence?: boolean;
  latency_ms?: number;
  receivedAt: number;
}

interface PartialMessageState {
  messageId?: string;
  revisions: PartialRevision[];
  truncated: boolean;
  sessionId?: string;
  speakerId?: string;
  source?: string;
}

interface AudioQueueItem {
  id: string;
  buffer: Buffer;
  textLen: number;
  receivedAt: number;
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// Track per-utterance message + history for edit-in-place
const partialMessages = new Map<string, PartialMessageState>(); // utteranceId -> state

// TTS Audio queue and player
const audioQueue: AudioQueueItem[] = [];
let isPlaying = false;
const audioPlayer = createAudioPlayer();

// Track player state
audioPlayer.on('stateChange', (oldState, newState) => {
  console.log(`[TTS] Player state: ${oldState.status} -> ${newState.status}`);

  if (newState.status === AudioPlayerStatus.Idle && oldState.status !== AudioPlayerStatus.Idle) {
    console.log(`[TTS] Finished playing, queue length: ${audioQueue.length}`);
    isPlaying = false;
    void processAudioQueue();
  }
});

audioPlayer.on('error', (error) => {
  console.error('[TTS] Player error:', error.message);
  isPlaying = false;
  void processAudioQueue();
});

async function processAudioQueue() {
  if (isPlaying || audioQueue.length === 0) {
    return;
  }

  const item = audioQueue.shift();
  if (!item) {
    return;
  }

  isPlaying = true;

  const queuedMs = Date.now() - item.receivedAt;
  console.log(`[TTS] Playing audio id=${item.id} bytes=${item.buffer.length} textLen=${item.textLen} queuedMs=${queuedMs} queueRemaining=${audioQueue.length}`);

  try {
    // Ensure voice connection is still active
    const connection = getVoiceConnection(GUILD_ID);
    if (!connection) {
      console.error('[TTS] No voice connection available');
      isPlaying = false;
      void processAudioQueue();
      return;
    }

    // Create audio resource from buffer
    const resource = createAudioResource(Readable.from(item.buffer), {
      inputType: StreamType.Arbitrary,
      metadata: { id: item.id }
    });

    audioPlayer.play(resource);
    console.log(`[TTS] Started playing id=${item.id} playerStatus=${audioPlayer.state.status}`);
  } catch (error) {
    console.error(`[TTS] Play error id=${item.id}:`, error);
    isPlaying = false;
    void processAudioQueue();
  }
}

// TTS HTTP Server
const ttsServer = http.createServer(async (req, res) => {
  const { method, url } = req;

  // Health check
  if (method === 'GET' && (url === '/health' || url === '/healthz')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      discord: {
        ready: client.isReady(),
        user: client.user?.tag
      },
      voice: {
        connected: getVoiceConnection(GUILD_ID) !== undefined
      },
      player: {
        status: audioPlayer.state.status,
        isPlaying,
        queueLen: audioQueue.length
      }
    }));
    return;
  }

  // TTS playback endpoint
  if (method === 'POST' && url === '/play') {
    const contentType = req.headers['content-type'];
    if (contentType !== 'audio/wav') {
      console.warn(`[TTS] Rejected invalid content-type: ${contentType}`);
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'content-type must be audio/wav' }));
      return;
    }

    const id = req.headers['x-speech-id'] as string || 'unknown';
    const bytes = req.headers['x-speech-bytes'] as string;
    const textLen = req.headers['x-speech-text-len'] as string;

    console.log(`[TTS] Receiving audio id=${id} expectedBytes=${bytes} textLen=${textLen}`);

    try {
      const chunks: Buffer[] = [];
      let total = 0;
      const MAX_SIZE = 10 * 1024 * 1024; // 10MB limit

      for await (const chunk of req) {
        const buf = Buffer.from(chunk);
        total += buf.length;
        if (total > MAX_SIZE) {
          res.writeHead(413, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'payload too large' }));
          return;
        }
        chunks.push(buf);
      }

      const buffer = Buffer.concat(chunks);
      console.log(`[TTS] Received audio id=${id} actualBytes=${buffer.length}`);

      const queueItem: AudioQueueItem = {
        id,
        buffer,
        textLen: textLen ? parseInt(textLen, 10) : 0,
        receivedAt: Date.now()
      };

      audioQueue.push(queueItem);
      console.log(`[TTS] Queued audio id=${id} position=${audioQueue.length}`);

      res.writeHead(202, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'queued',
        id,
        position: audioQueue.length
      }));

      void processAudioQueue();
    } catch (error) {
      const err = error as Error;
      console.error(`[TTS] Receive error id=${id}:`, err.message);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

ttsServer.listen(TTS_PLAYER_PORT, () => {
  console.log(`[TTS] HTTP server listening on port ${TTS_PLAYER_PORT}`);
});

ttsServer.on('error', (err) => {
  console.error('[TTS] Server error:', err);
});

function isPartialEvent(ev: OutEvent): ev is PartialEventPayload {
  return ev.type === 'partial';
}

function isFinalEvent(ev: OutEvent): ev is FinalEventPayload {
  return ev.type === 'final';
}

function formatConfidence(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : 'n/a';
}

function formatRevisionLine(revision: PartialRevision): string {
  const segments = [
    `${revision.revision}`,
    `${revision.t0_ms}-${revision.t1_ms}ms`,
    `conf ${formatConfidence(revision.confidence)}`,
  ];
  if (typeof revision.latency_ms === 'number') {
    segments.push(`lat ${revision.latency_ms}ms`);
  }
  const suffix = revision.lowConfidence ? ' [low confidence]' : '';
  return `(${segments.join(' | ')}) ${JSON.stringify(revision.text)}${suffix}`;
}

function shortenLine(line: string, max = 180): string {
  if (line.length <= max) return line;
  const limit = Math.max(0, max - 3);
  return `${line.slice(0, limit)}...`;
}

function formatDebugBlock(obj: Record<string, unknown>, pretty: boolean): string {
  const payload = pretty ? JSON.stringify(obj, null, 2) : JSON.stringify(obj);
  return ['```json', payload, '```'].join('\n');
}

function pushRevision(state: PartialMessageState, ev: PartialEventPayload): void {
  state.sessionId = ev.session_id || state.sessionId;
  state.speakerId = ev.speaker_id || state.speakerId;
  state.source = ev.source || state.source;
  state.revisions.push({
    revision: ev.revision,
    text: typeof ev.text === 'string' ? ev.text : '',
    t0_ms: ev.t0_ms,
    t1_ms: ev.t1_ms,
    confidence: ev.confidence,
    lowConfidence: ev.low_confidence === true,
    latency_ms: typeof ev.latency_ms === 'number' ? ev.latency_ms : undefined,
    receivedAt: Date.now(),
  });
}

function buildPartialContent(key: string, state: PartialMessageState, ev: PartialEventPayload): string {
  while (true) {
    const last = state.revisions[state.revisions.length - 1];
    const metaParts = [
      `session ${state.sessionId ?? ev.session_id ?? '-'}`,
      `speaker ${state.speakerId ?? ev.speaker_id ?? '-'}`,
      `source ${state.source ?? ev.source ?? 'whisper'}`,
      `revisions ${state.revisions.length}`,
    ];
    if (last && typeof last.latency_ms === 'number') {
      metaParts.push(`last lat ${last.latency_ms}ms`);
    }
    if (last?.lowConfidence) {
      metaParts.push('last low confidence');
    }
    const header = `Partial updates (${key})`;
    const parts: string[] = [header, metaParts.join(' | ')];
    if (state.revisions.length) {
      parts.push('');
      if (state.truncated) {
        parts.push(PARTIAL_TRUNCATION_NOTICE);
      }
      parts.push(...state.revisions.map((revision) => formatRevisionLine(revision)));
    }
    const content = parts.join('\n');
    if (content.length <= DISCORD_MESSAGE_LIMIT) {
      return content;
    }
    if (state.revisions.length > 1) {
      state.revisions.shift();
      state.truncated = true;
      continue;
    }
    return `${content.slice(0, DISCORD_MESSAGE_LIMIT - MESSAGE_TRIM_NOTICE.length - 1)}\n${MESSAGE_TRIM_NOTICE}`;
  }
}

function buildFinalDebug(ev: FinalEventPayload, historyCount: number, historyTruncated: boolean): Record<string, unknown> {
  const debug: Record<string, unknown> = {
    session_id: ev.session_id,
    speaker_id: ev.speaker_id,
    utterance_id: ev.utterance_id,
    history_revisions: historyCount,
    history_truncated: historyTruncated,
    t0_ms: ev.t0_ms,
    t1_ms: ev.t1_ms,
    frames: ev.frames,
    voiced_ms: ev.voiced_ms,
    silence_tail_ms: ev.silence_tail_ms,
    dropped_frames: ev.dropped_frames,
    source: ev.source,
    confidence: ev.confidence,
  };
  if (typeof ev.latency_ms === 'number') {
    debug.latency_ms = ev.latency_ms;
  }
  return debug;
}

function buildFinalContent(key: string, state: PartialMessageState | undefined, ev: FinalEventPayload): string {
  const historyLines = state ? state.revisions.map((revision) => formatRevisionLine(revision)) : [];
  let workingHistory = [...historyLines];
  let historyTruncated = state?.truncated ?? false;
  const header = `Final transcript (${key})`;
  const metaBase = [
    `session ${state?.sessionId ?? ev.session_id ?? '-'}`,
    `speaker ${state?.speakerId ?? ev.speaker_id ?? '-'}`,
    `source ${state?.source ?? ev.source ?? 'whisper'}`,
  ];
  let usePrettyDebug = true;

  while (true) {
    const metaParts = [...metaBase, `revisions ${workingHistory.length}`];
    if (typeof ev.latency_ms === 'number') {
      metaParts.push(`lat ${ev.latency_ms}ms`);
    }
    const finalLine = `Final text: ${JSON.stringify(typeof ev.text === 'string' ? ev.text : '')}`;
    const debugBlock = formatDebugBlock(buildFinalDebug(ev, workingHistory.length, historyTruncated), usePrettyDebug);
    const lines: string[] = [header, metaParts.join(' | '), '', finalLine];
    if (workingHistory.length) {
      lines.push('');
      lines.push('**Partial history**');
      if (historyTruncated) {
        lines.push(`> ${PARTIAL_TRUNCATION_NOTICE}`);
      }
      lines.push(...workingHistory.map((line) => `> ${line}`));
    }
    lines.push('');
    lines.push(debugBlock);
    const content = lines.join('\n').trim();
    if (content.length <= DISCORD_MESSAGE_LIMIT) {
      return content;
    }
    if (workingHistory.length > 1) {
      workingHistory = workingHistory.slice(1);
      historyTruncated = true;
      continue;
    }
    if (workingHistory.length === 1) {
      workingHistory = [shortenLine(workingHistory[0])];
      historyTruncated = true;
      continue;
    }
    if (usePrettyDebug) {
      usePrettyDebug = false;
      continue;
    }
    return `${content.slice(0, DISCORD_MESSAGE_LIMIT - MESSAGE_TRIM_NOTICE.length - 1)}\n${MESSAGE_TRIM_NOTICE}`;
  }
}

function ensureJoined(): VoiceConnection {
  const existing = getVoiceConnection(GUILD_ID);
  if (existing) return existing;
  return joinVoiceChannel({
    channelId: VOICE_CHANNEL_ID,
    guildId: GUILD_ID,
    adapterCreator: client.guilds.cache.get(GUILD_ID)!.voiceAdapterCreator,
    selfDeaf: false,
    decryptionFailureTolerance: 50, // Tolerate some DAVE decryption failures without disconnecting
  });
}

function makePipeline(vc: VoiceConnection, userId: string, onEvent: (ev: OutEvent) => void) {
  const receiver = vc.receiver;
  const opus = receiver.subscribe(userId, { end: { behavior: EndBehaviorType.AfterSilence, duration: 1000 } });
  const opusDecoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 }); // 20ms @48k
  const ff = spawn(ffmpeg as string, [
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    's16le',
    '-ar',
    '48000',
    '-ac',
    '2',
    '-i',
    'pipe:0',
    '-f',
    's16le',
    '-ar',
    '16000',
    '-ac',
    '1',
    'pipe:1',
  ]);
  opus.pipe(opusDecoder).pipe(ff.stdin);
  const chunker = new FrameChunker();
  ff.stdout.pipe(chunker);

  // AudioGate: only create session when real voice is detected
  const gate = new AudioGate({
    minFrames: 3,        // 60ms minimum
    minVoicedFrames: 2,  // At least 2 frames with voice
    rmsThreshold: 0.015, // RMS threshold for voice detection (lowered for quieter speech)
    debug: process.env.AUDIO_GATE_DEBUG === '1',
  });

  let session: WSSession | null = null;
  let sessionConnecting = false;
  let sessionReady = false;
  let pendingFrames: Buffer[] = [];  // Buffer frames while session connects

  gate.on('open', () => {
    // Voice detected - create and connect session
    if (session || sessionConnecting) return;
    sessionConnecting = true;
    
    const newSession = new WSSession(WS_URL, { sessionId: randomUUID(), speakerId: userId });
    newSession
      .connect(onEvent)
      .then(() => {
        session = newSession;
        sessionConnecting = false;
        sessionReady = true;
        
        // Flush all frames that arrived while session was connecting
        const framesToFlush = pendingFrames;
        pendingFrames = [];
        for (const frame of framesToFlush) {
          session.writeFrame(frame);
        }
        
        if (process.env.AUDIO_GATE_DEBUG === '1') {
          console.log(`[AudioGate] Session ready for user=${userId}, flushed ${framesToFlush.length} pending frames`);
        }
      })
      .catch((err) => {
        console.error('[AudioGate] ws connect error', err);
        sessionConnecting = false;
        pendingFrames = []; // Discard on error
      });
  });

  gate.on('frame', (frame: Buffer) => {
    if (sessionReady && session) {
      // Session is connected, send immediately
      session.writeFrame(frame);
    } else {
      // Session still connecting, buffer the frame
      pendingFrames.push(frame);
    }
  });

  gate.on('close', () => {
    // Gate closed - end session if it exists
    if (session) {
      session.end();
    }
    pendingFrames = [];
  });

  gate.on('discard', ({ frames, reason }: { frames: number; reason: string }) => {
    console.log(`[AudioGate] Discarded ${frames} frames (${reason}) for user=${userId} - no session created`);
    pendingFrames = [];
  });

  // Route chunker output through the gate
  chunker.on('data', (frame: Buffer) => {
    gate.push(frame);
  });

  chunker.on('close', () => {
    gate.end();
  });

  opus.on('end', () => {
    try {
      ff.stdin.end();
    } catch {
      /* noop */
    }
  });

  ff.on('close', () => {
    gate.end();
  });

  return {
    session: null as WSSession | null, // Session is created lazily
    gate,
    cleanup: () => {
      try {
        opus.destroy();
      } catch {
        /* noop */
      }
      try {
        ff.kill('SIGKILL');
      } catch {
        /* noop */
      }
      gate.end();
    },
  };
}

async function postOrEditPartial(channel: TextChannel, ev: PartialEventPayload) {
  if (!PARTIALS_ENABLE) return;

  const key = ev.utterance_id;
  let state = partialMessages.get(key);
  if (!state) {
    state = { revisions: [], truncated: false };
    partialMessages.set(key, state);
  }

  pushRevision(state, ev);
  const content = buildPartialContent(key, state, ev);

  if (!state.messageId) {
    const msg = await channel.send(content);
    state.messageId = msg.id;
    return;
  }

  try {
    const msg = await channel.messages.fetch(state.messageId);
    await msg.edit(content);
  } catch {
    const msg = await channel.send(content);
    state.messageId = msg.id;
  }
}

async function replaceWithFinal(channel: TextChannel, ev: FinalEventPayload) {
  const key = ev.utterance_id;
  const state = partialMessages.get(key);
  const content = buildFinalContent(key, state, ev);

  if (state?.messageId) {
    try {
      const msg = await channel.messages.fetch(state.messageId);
      await msg.edit(content);
    } catch {
      await channel.send(content);
    }
  } else {
    await channel.send(content);
  }

  partialMessages.delete(key);
}

async function handleOutEvent(channel: TextChannel, finalsChannel: TextChannel | null, ev: OutEvent) {
  if (isPartialEvent(ev)) {
    await postOrEditPartial(channel, ev);
    return;
  }
  if (isFinalEvent(ev)) {
    const eventTimestamp = new Date();
    await replaceWithFinal(channel, ev);
    await sendFinalSimpleMessage(finalsChannel, ev, DISCORD_MESSAGE_LIMIT, () => eventTimestamp);
    return;
  }
  if (ev.type === 'error') {
    console.warn('server error', ev);
  }
}

client.once(Events.ClientReady, async () => {
  console.log(`Logged in as ${client.user?.tag}`);
  const channel = (await client.channels.fetch(TRANSCRIPT_CHANNEL_ID)) as TextChannel;
  let finalsChannel: TextChannel | null = null;
  if (FINALS_TRANSCRIPT_CHANNEL_ID) {
    try {
      const fetched = await client.channels.fetch(FINALS_TRANSCRIPT_CHANNEL_ID);
      if (fetched && fetched.isTextBased()) {
        finalsChannel = fetched as TextChannel;
      }
    } catch (err) {
      console.warn('failed to fetch finals transcript channel', err);
    }
  }
  // Start the WebSocket forwarder for external STT consumers
  startForwarder();

  const vc = ensureJoined();

  // Subscribe audio player to voice connection for TTS playback
  vc.subscribe(audioPlayer);
  console.log('[TTS] Audio player subscribed to voice connection');

  const active = new Map<string, ReturnType<typeof makePipeline>>();

  vc.receiver.speaking.on('start', async (userId) => {
    if (active.has(userId)) return; // prevent dup
    const pipeline = makePipeline(vc, userId, (ev) => {
      void handleOutEvent(channel, finalsChannel, ev);
      broadcast(ev);
    });
    active.set(userId, pipeline);
  });

  vc.receiver.speaking.on('end', (userId) => {
    const pipeline = active.get(userId);
    if (!pipeline) return;
    pipeline.cleanup();
    active.delete(userId);
  });
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('[TTS] Shutting down...');
  ttsServer.close();
  client.destroy();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('[TTS] Shutting down...');
  ttsServer.close();
  client.destroy();
  process.exit(0);
});

// Track decryption failures for rate-limited notifications
let lastDecryptionNotification = 0;
const DECRYPTION_NOTIFICATION_COOLDOWN_MS = 60_000; // 1 minute cooldown

async function notifyDecryptionFailure(error: Error) {
  const now = Date.now();
  if (now - lastDecryptionNotification < DECRYPTION_NOTIFICATION_COOLDOWN_MS) {
    console.warn('[DAVE] Decryption failure (notification throttled):', error.message);
    return;
  }
  lastDecryptionNotification = now;

  console.warn('[DAVE] Decryption failure:', error.message);

  if (!client.isReady()) {
    console.warn('[DAVE] Cannot send notification - client not ready');
    return;
  }

  try {
    const channel = await client.channels.fetch(TRANSCRIPT_CHANNEL_ID);
    if (channel?.isTextBased()) {
      const textChannel = channel as TextChannel;
      const timestamp = new Date().toISOString();
      await textChannel.send(
        `[DAVE Warning] Decryption failure at ${timestamp}\n` +
        `\`\`\`\n${error.message}\n\`\`\`\n` +
        `This may occur when users with older Discord clients join. The bot will continue operating.`
      );
    }
  } catch (notifyErr) {
    console.error('[DAVE] Failed to send notification:', notifyErr);
  }
}

// Handle DAVE decryption failures gracefully
process.on('uncaughtException', (error) => {
  if (error.message?.includes('DecryptionFailed') || error.message?.includes('decrypt')) {
    void notifyDecryptionFailure(error);
    return; // Don't crash - these are non-fatal
  }
  // For other errors, log and exit
  console.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  const error = reason instanceof Error ? reason : new Error(String(reason));
  if (error.message?.includes('DecryptionFailed') || error.message?.includes('decrypt')) {
    void notifyDecryptionFailure(error);
    return; // Don't crash
  }
  console.error('Unhandled rejection:', reason);
});

client.login(TOKEN);
