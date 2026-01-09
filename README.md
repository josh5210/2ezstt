# EZSTT: Real-Time Speech-to-Text System

**Real-time speech-to-text for Discord voice channels.** Stream audio → transcribe with local Whisper or OpenAI API → get live partial + final transcripts via WebSocket.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- **For local Whisper**: Download a [faster-whisper model](https://huggingface.co/Systran/faster-whisper-medium)
- **For OpenAI**: Get an [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Install Dependencies

```bash
# Python (STT server)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Node (Discord client)
npm install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

**Minimal config for local Whisper:**
```env
EZSTT_BACKEND=whisper
EZSTT_MODEL_DIR=/path/to/faster-whisper-medium
```

**Minimal config for OpenAI:**
```env
EZSTT_BACKEND=openai
OPENAI_API_KEY=sk-...
```

**Discord bot (optional):**
```env
DISCORD_TOKEN=your-bot-token
GUILD_ID=...
VOICE_CHANNEL_ID=...
TRANSCRIPT_CHANNEL_ID=...
```

### 3. Run

```bash
# Terminal 1: Start STT server
python -m uvicorn sidecar.server:app --host 127.0.0.1 --port 8767

# Terminal 2: Start Discord client (if using Discord)
npm run dev
```

### 4. Connect

Send audio to `ws://127.0.0.1:8767/ws` and receive `partial` and `final` transcript events.

---

## Discord Bot Setup

### Create a Bot Application
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** → Name it → **Create**
3. Go to **Bot** → Click **Add Bot**
4. Copy the **Token** → paste into `.env` as `DISCORD_TOKEN`

### Required Bot Permissions
Go to **OAuth2 → URL Generator** and select:

**Scopes:**
- ✅ `bot`

**Bot Permissions:**
- ✅ `Connect` — Join voice channels
- ✅ `Speak` — Play TTS audio (optional)
- ✅ `View Channels` — See channels
- ✅ `Send Messages` — Post transcripts
- ✅ `Read Message History` — Edit partial messages

### Invite to Server
1. Copy the generated OAuth2 URL
2. Open in browser → Select your server → Authorize

### Get Channel IDs
1. Enable **Developer Mode** in Discord (User Settings → Advanced)
2. Right-click channels → **Copy ID**
3. Add to `.env`:
   ```env
   GUILD_ID=your-server-id
   VOICE_CHANNEL_ID=voice-channel-to-listen
   TRANSCRIPT_CHANNEL_ID=text-channel-for-transcripts
   ```

---

## Architecture


## System Overview

EZSTT is a dual-component system consisting of a Python-based STT server (`sidecar`) and a TypeScript Discord client. The architecture is designed around WebSocket-based streaming audio processing with clear separation between audio capture, speech processing, and output formatting.

### Core Components

#### 1. Discord Audio Client (`client/`)
- **Primary Entry**: `discord_stream.ts` - Main Discord bot orchestrating voice capture and transcription display
- **Audio Processing**: `audio/frame_chunker.ts` - Converts streaming audio into fixed-size frames (320 bytes/20ms)
- **Voice Activity Gate**: `audio/audio_gate.ts` - Client-side voice detection to prevent spurious sessions
- **WebSocket Interface**: `ws_session.ts` - Manages STT server communication with session lifecycle
- **Output Formatting**: `finals_channel.ts` - Handles clean final transcript delivery to designated channels

#### 2. STT Processing Server (`sidecar/`)
- **Core Engine**: `server.py` - FastAPI WebSocket server handling concurrent transcription sessions
- **Speech Detection**: `endpoint.py` - WebRTC VAD-based speech endpointing with ring buffer
- **STT Backend**: `whisper_backend.py` - faster-whisper integration with configurable models
- **Partial Results**: `partials.py` - Streaming partial transcription engine for real-time feedback
- **Keyword Detection**: `kws_vosk.py` - Optional keyword spotting for trigger-based workflows

## Audio Processing Pipeline

### Input Path (Discord → STT Server)
```
Discord Voice Channel
│
├─ Opus Stream (48kHz stereo)
│  └─ prism-media decoder
│     └─ FFmpeg resample (16kHz mono PCM16)
│        └─ FrameChunker (320 bytes/20ms frames)
│           └─ AudioGate (voice activity detection)
│              ├─ Discarded (no voice detected)
│              └─ Voice confirmed → WebSocket session created
│                 └─ Binary frames → STT Server endpoint
```

### AudioGate: Client-Side Voice Detection

The `AudioGate` prevents spurious STT sessions by buffering audio locally and only creating WebSocket connections when actual speech is detected. This is a critical optimization that:

- **Reduces GPU load**: No transcription attempts on silence/noise
- **Reduces network traffic**: No WebSocket connections until voice confirmed
- **Preserves sentence beginnings**: Rolling buffer includes pre-detection audio

```
┌─────────────────────────────────────────────────────────┐
│                     AudioGate Flow                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Frame arrives ──► Buffer (up to 50 frames)             │
│                         │                                │
│                         ▼                                │
│               Calculate RMS Energy                       │
│                         │                                │
│            ┌────────────┴────────────┐                   │
│            ▼                         ▼                   │
│     RMS < threshold           RMS ≥ threshold            │
│     (silence/noise)           (voiced frame)             │
│            │                         │                   │
│            ▼                         ▼                   │
│    Continue buffering         Increment voiced count     │
│            │                         │                   │
│            ▼                         ▼                   │
│   ┌────────────────┐      ┌─────────────────────┐       │
│   │ Max buffer or  │      │ minVoicedFrames met │       │
│   │ timeout reached│      │ within minFrames?   │       │
│   └───────┬────────┘      └──────────┬──────────┘       │
│           ▼                          ▼                   │
│     DISCARD buffer            OPEN gate                  │
│     (no session created)      ├─ Emit 'open' event       │
│                               ├─ Create WebSocket        │
│                               ├─ Flush buffered frames   │
│                               └─ Stream new frames       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Hardware/Network Impact Summary:**

| Resource | Impact | Details |
|----------|--------|---------|
| Client GPU | None | No ML on client; pure CPU/memory |
| Server GPU | **↓ Reduced** | Fewer spurious sessions = fewer inference attempts |
| Network | **↓ Reduced** | No traffic until voice confirmed |
| Client Memory | +32KB/speaker | Circular buffer (50 frames × 640 bytes) |
| Client CPU | +0.001ms/frame | RMS calculation is trivial |

### STT Processing Flow
```
WebSocket Frame Input
│
├─ Ring Buffer (preroll storage)
│
├─ WebRTC VAD (voice activity detection)
│  ├─ IDLE state (collecting preroll)
│  └─ IN_SPEECH state (building utterance)
│
├─ Speech Endpoint Detection
│  └─ Utterance Assembly (audio + metadata)
│
├─ Whisper Processing
│  ├─ Partial Results Engine (streaming updates)
│  └─ Final Transcription
│
└─ WebSocket JSON Events
   ├─ partial events (real-time updates)
   └─ final events (complete transcripts)
```

### Output Path (STT Server → Discord)
```
STT Server Events
│
├─ Partial Events (type: 'partial')
│  ├─ utterance_id (unique per speech segment)
│  ├─ revision (incremental updates)
│  ├─ text (current transcription)
│  ├─ confidence score
│  └─ timing metadata
│
├─ Final Events (type: 'final')
│  ├─ utterance_id (matches partials)
│  ├─ text (final transcription)
│  ├─ confidence score
│  └─ processing metrics
│
└─ Discord Message Management
   ├─ Live editing of partial messages
   ├─ Final replacement with complete transcript
   └─ Optional clean finals-only channel
```

## Integration Points & Hooks

### WebSocket Message Protocol

#### Session Management
```json
// Client → Server: Start session
{
  "type": "session.start",
  "session_id": "uuid",
  "speaker_id": "discord_user_id",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "transport": "binary",
  "meta": {"app": "ezstt-client"}
}

// Client → Server: End session
{
  "type": "session.end"
}
```

#### STT Event Stream
```json
// Server → Client: Partial transcript
{
  "type": "partial",
  "session_id": "uuid",
  "speaker_id": "discord_user_id",
  "utterance_id": "uuid",
  "revision": 3,
  "text": "hello world",
  "t0_ms": 1234,
  "t1_ms": 2345,
  "confidence": 0.95,
  "latency_ms": 150,
  "low_confidence": false,
  "source": "whisper"
}

// Server → Client: Final transcript
{
  "type": "final",
  "session_id": "uuid",
  "speaker_id": "discord_user_id",
  "utterance_id": "uuid",
  "text": "hello world everyone",
  "confidence": 0.98,
  "source": "whisper",
  "t0_ms": 1234,
  "t1_ms": 3456,
  "frames": 65,
  "voiced_ms": 1800,
  "silence_tail_ms": 250,
  "latency_ms": 180
}
```

### Configuration Interfaces

#### Environment Variables (Client)
- `DISCORD_TOKEN` - Discord bot authentication
- `VOICE_CHANNEL_ID` - Target voice channel for listening
- `TRANSCRIPT_CHANNEL_ID` - Channel for detailed transcripts with partial history
- `FINALS_TRANSCRIPT_CHANNEL_ID` - Optional clean finals-only channel
- `EZSTT_WS_URL` - STT server WebSocket endpoint
- `PARTIALS_ENABLE` - Toggle partial transcript display (default: enabled)
- `AUDIO_GATE_DEBUG` - Enable verbose AudioGate logging (default: disabled, set to `1` to enable)

#### Environment Variables (Server)
- `EZSTT_MODEL` - Whisper model size (default: "medium")
- `EZSTT_DEVICE` - Processing device ("cpu", "cuda", "auto")
- `EZSTT_LANG` - Target language for transcription (default: "en")
- `EZSTT_BEAM_SIZE` - Whisper beam search size (default: 1)
- `EZSTT_PAD_SHORT_MS` - Minimum utterance padding (default: 450ms)
- `EZSTT_MAX_UTTER_MS` - Maximum utterance length (default: 30s)
- `EZSTT_LOG_LEVEL` - Server log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

### Extension Points for Future AI Integration

#### Speaker Identification Hooks
- `speaker_id` field in all events maps to Discord user IDs
- Can be extended to support voice biometric identification
- Enables per-speaker conversation tracking for AI character context

#### Conversation Context Integration
- `session_id` provides conversation session boundaries
- `utterance_id` enables utterance-level conversation threading
- Timing metadata (`t0_ms`, `t1_ms`) supports turn-taking analysis
- Confidence scoring enables quality filtering for AI input

#### TTS Integration Points
- WebSocket event stream can trigger TTS responses
- `latency_ms` metrics enable end-to-end latency optimization
- Finals-only channel provides clean input for AI character responses
- Session lifecycle hooks support conversation state management

#### Keyword/Intent Detection Extensions
- `kws_vosk.py` provides grammar-based keyword spotting foundation
- Events carry `source` field to distinguish STT engines
- Modular backend design supports multiple concurrent STT engines
- Confidence thresholding enables intent classification triggers

#### AI Character Response Hooks
```json
// Future: AI character trigger events
{
  "type": "intent_detected",
  "session_id": "uuid",
  "speaker_id": "user_123",
  "utterance_id": "uuid",
  "intent": "question_about_weather",
  "entities": {"location": "seattle"},
  "confidence": 0.89,
  "response_required": true
}

// Future: TTS request integration
{
  "type": "tts_request",
  "character_id": "assistant_bot",
  "response_to": "utterance_uuid",
  "text": "The weather in Seattle is cloudy today",
  "voice_settings": {"speed": 1.0, "pitch": 0.0}
}
```

## Debugging & Troubleshooting

### Debug Environment Variables

Enable verbose logging for troubleshooting:

```bash
# Client-side: AudioGate voice detection logs
AUDIO_GATE_DEBUG=1

# Server-side: Full debug logging (session lifecycle, VAD, transcription)
EZSTT_LOG_LEVEL=DEBUG
```

### Debug Output Examples

**AudioGate Debug** (`AUDIO_GATE_DEBUG=1`):
```
[AudioGate] frame=1 buffered=1 voiced=0 rms=0.0012
[AudioGate] frame=2 buffered=2 voiced=0 rms=0.0018
[AudioGate] frame=3 buffered=3 voiced=1 rms=0.0234
[AudioGate] frame=4 buffered=4 voiced=2 rms=0.0456
[AudioGate] OPEN reason=voice_detected buffered=4 voiced=2
[AudioGate] Session ready for user=123456789, flushed 4 pending frames
```

**Server Debug** (`EZSTT_LOG_LEVEL=DEBUG`):
```
[Session abc123] LIFECYCLE: created -> handshake_complete
[Session abc123] RX binary: 640 bytes (total: 1 binary, 0 json)
[Endpoint abc123] VAD state: IDLE -> IN_SPEECH (frame=5)
[Endpoint abc123] stats: frames=50 voiced=38 transitions=2
[Session abc123] LIFECYCLE: active -> closing:peer_disconnect
```

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Sessions with 0 frames | Client creating sessions before audio flows | AudioGate should prevent this; check `AUDIO_GATE_DEBUG` |
| Clipped sentence beginnings | RMS threshold too high or buffer too small | Lower `rmsThreshold` or increase `minFrames` |
| Too many spurious sessions | RMS threshold too low | Raise `rmsThreshold` |
| No transcriptions | WebSocket not connecting | Check `EZSTT_WS_URL` and server status |
| High latency | Model too large or CPU-only | Use smaller model or enable CUDA |

### AudioGate Tuning Parameters

Adjust in `client/discord_stream.ts` within `makePipeline()`:

```typescript
const gate = new AudioGate({
  minFrames: 3,        // Minimum frames before checking voice (60ms)
  minVoicedFrames: 2,  // Required voiced frames to trigger
  rmsThreshold: 0.015, // RMS energy threshold (0.0-1.0)
  maxBufferFrames: 50, // Max buffer before force-discard (1 second)
  bufferTimeoutMs: 2000, // Timeout for stale buffers
  debug: true,         // Enable logging
});
```

**Tuning Guidelines:**

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `rmsThreshold` | More sensitive (catches quiet speech) | Less sensitive (filters more noise) |
| `minVoicedFrames` | Faster triggering | More confirmation required |
| `minFrames` | Less buffered context | More buffered context preserved |
| `maxBufferFrames` | Less memory, faster discard | More tolerance for delayed speech |

**Recommended starting values:**
- Quiet environment: `rmsThreshold: 0.01`
- Noisy environment: `rmsThreshold: 0.03`
- Quick speakers: `minVoicedFrames: 1`
- Deliberate speakers: `minVoicedFrames: 3`

---

## Technical Architecture

### Audio Frame Processing
- **Frame Size**: 320 bytes (20ms @ 16kHz mono PCM16)
- **Client-Side Gate**: AudioGate buffers up to 50 frames (1 second) while detecting voice
- **Server-Side Buffer**: Ring buffer maintains 120ms preroll for utterance context
- **Voice Detection**: 
  - Client: RMS energy threshold (fast, prevents spurious sessions)
  - Server: WebRTC VAD with configurable aggressiveness (0-3)
- **Endpointing**: Minimum 120ms speech, 250ms silence for utterance boundaries

### STT Engine Configuration
- **Model Backend**: faster-whisper with configurable model sizes
- **Processing**: Single-utterance transcription with beam search
- **Partial Generation**: Streaming updates every 300ms during speech
- **Quality Control**: Confidence scoring and low-confidence flagging

### Performance Characteristics
- **Latency**: ~150-300ms end-to-end for partial results
- **Final Latency**: ~500-1000ms depending on utterance length and model size
- **Throughput**: Supports concurrent multi-speaker processing
- **Memory**: O(1) per-session with bounded audio buffers

### WebSocket Session Management
- **Connection Lifecycle**: Automatic reconnection and cleanup
- **Backpressure Handling**: Frame dropping under high load (200 frame queue limit)
- **Session Isolation**: Independent processing per speaker/session
- **Error Recovery**: Graceful handling of connection drops and server restarts

## Message Flow Examples

### Typical Conversation Flow
```
1. User starts speaking in Discord voice channel
   → Discord client receives audio via Opus stream
   → Audio pipeline: Opus → PCM → FrameChunker → AudioGate

2. AudioGate detects voice activity
   → Buffers initial frames while checking RMS energy
   → Voice confirmed: emits 'open' event
   → WebSocket session created
   → Buffered frames flushed to server

3. STT server receives frames
   → VAD confirms speech (server-side)
   → Ring buffer provides preroll context
   → Whisper processing begins

4. Streaming partial results (every ~300ms)
   → partial events with incremental revisions
   → Discord message live-editing
   → Confidence and timing metadata

5. User stops speaking
   → Silence detection triggers endpoint
   → Final Whisper processing
   → final event replaces partial message
   → Optional clean final posted to separate channel
   → AudioGate closes, session ends
```

### Multi-Speaker Handling
```
Speaker A: "Hey everyone, how's the weather?"
├─ session_id: session_123, speaker_id: user_456
├─ partial events: "Hey everyone" → "Hey everyone how's" → "Hey everyone how's the weather"
└─ final event: "Hey everyone, how's the weather?"

Speaker B: "It's raining here in Seattle"
├─ session_id: session_123, speaker_id: user_789
├─ partial events: "It's raining" → "It's raining here" → "It's raining here in Seattle"
└─ final event: "It's raining here in Seattle"
```

## Future Integration Scenarios

### AI Character Integration
The system is designed to support AI character workflows where STT feeds conversation context:
- **Real-time Context**: Partial events enable AI to prepare responses during user speech
- **Speaker Tracking**: Per-user conversation history and response personalization
- **Intent Detection**: Keyword spotting and confidence scoring enable smart response triggers
- **Latency Optimization**: Streaming architecture minimizes response delays

### TTS Pipeline Integration
WebSocket event streams can directly trigger TTS synthesis:
- **Response Timing**: Use silence detection and final events to time AI responses
- **Voice Cloning Context**: Speaker IDs enable per-character voice synthesis
- **Conversation Flow**: Session management supports multi-turn dialogue state

### Multi-Modal Extensions
Architecture supports expansion beyond voice:
- **Text Input**: WebSocket protocol can handle text message events
- **Visual Context**: Session metadata can include video/screen sharing context
- **Environmental Audio**: Non-speech audio classification for ambient awareness

## Testing & Development

The system includes comprehensive testing coverage:
- **Unit Tests**: Frame processing, WebSocket session management, STT integration
- **Integration Tests**: End-to-end audio pipeline validation
- **Performance Tests**: Latency benchmarking and concurrent session testing
- **GPU Smoke Tests**: Hardware acceleration validation

Development tools and utilities are provided for debugging and optimization of the real-time processing pipeline.

## Deployment Considerations

### Resource Requirements
- **CPU**: Multi-threaded Whisper processing (configurable thread count)
- **GPU**: Optional CUDA acceleration for larger models
- **Memory**: ~1-4GB depending on model size and concurrent sessions
- **Network**: Low-latency WebSocket connectivity for real-time performance

### Scalability Patterns
- **Horizontal**: Multiple STT server instances behind load balancer
- **Vertical**: Larger Whisper models for improved accuracy
- **Hybrid**: CPU partials + GPU finals for latency/accuracy optimization

This architecture provides a solid foundation for building sophisticated conversational AI systems that require real-time speech understanding with tight integration between STT, AI reasoning, and TTS synthesis components.