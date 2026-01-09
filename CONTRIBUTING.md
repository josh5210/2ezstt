# Contributing to EZSTT

Thank you for your interest in contributing to EZSTT! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ezstt.git`
3. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

### Python Sidecar (STT Server)
```bash
cd ezstt
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### TypeScript Client
```bash
npm install
```

## Code Style

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Use strict TypeScript, follow existing patterns

## Testing

Run tests before submitting:

```bash
# Python tests
python -m pytest tests/ -v

# TypeScript (if applicable)
npm test
```

## Debugging

### Enable Debug Logging

For development and troubleshooting, enable verbose logging:

```bash
# Client-side: AudioGate voice detection debugging
export AUDIO_GATE_DEBUG=1

# Server-side: Full debug logging
export EZSTT_LOG_LEVEL=DEBUG
```

### Key Debug Points

**AudioGate (`client/audio/audio_gate.ts`)**
- Logs every frame with RMS energy calculation
- Logs gate open/close events with reason
- Shows buffer state and voiced frame counts

**Server Session (`sidecar/server.py`)**
- Session lifecycle state transitions
- Message counts (binary vs JSON)
- Frame timing (first/last frame timestamps)
- Warnings for sessions ending with 0 frames

**Endpoint/VAD (`sidecar/endpoint.py`)**
- VAD state transitions (IDLE ↔ IN_SPEECH)
- Frame statistics at key intervals
- Voiced/unvoiced frame ratios

**Whisper Backend (`sidecar/whisper_backend.py`)**
- Model load timing and device selection
- Transcription events with duration and confidence
- Device fallback attempts

### Debug Output Interpretation

```
# Good: Voice detected, session created with frames
[AudioGate] OPEN reason=voice_detected buffered=4 voiced=2
[Session x] RX binary: 640 bytes (total: 50 binary, 0 json)

# Bad: Session created but no audio received
[Session x] WARNING: session ending with 0 frames received

# Bad: Too many discards (threshold may be too high)
[AudioGate] RESET reason=timeout discarding=50 frames
```

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Add a clear PR description
4. Wait for review

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Include environment details (OS, Python version, Node version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
