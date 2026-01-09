import { EventEmitter } from 'node:events';

export interface AudioGateConfig {
  /** Minimum frames to buffer before checking for voice (default: 3 = 60ms) */
  minFrames?: number;
  /** Minimum voiced frames required to trigger (default: 2) */
  minVoicedFrames?: number;
  /** RMS threshold for considering a frame "voiced" (default: 0.02) */
  rmsThreshold?: number;
  /** Max frames to buffer before forcing open (default: 50 = 1s) */
  maxBufferFrames?: number;
  /** Timeout to discard stale buffer if no voice detected (default: 2000ms) */
  bufferTimeoutMs?: number;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

const DEFAULT_CONFIG: Required<AudioGateConfig> = {
  minFrames: 3,
  minVoicedFrames: 2,
  rmsThreshold: 0.02,
  maxBufferFrames: 50,
  bufferTimeoutMs: 2000,
  debug: false,
};

/**
 * AudioGate buffers incoming PCM16 frames and only opens (emits 'open' + buffered frames)
 * when it detects sufficient voice activity based on RMS energy.
 * 
 * Events:
 * - 'open': Emitted once when voice is detected, before any frames
 * - 'frame': Emitted for each frame (including buffered) after gate opens
 * - 'close': Emitted when the gate is closed/reset
 * 
 * Usage:
 *   const gate = new AudioGate();
 *   gate.on('open', () => createSession());
 *   gate.on('frame', (frame) => session.writeFrame(frame));
 *   chunker.on('data', (frame) => gate.push(frame));
 *   chunker.on('close', () => gate.end());
 */
export class AudioGate extends EventEmitter {
  private config: Required<AudioGateConfig>;
  private buffer: Buffer[] = [];
  private voicedCount = 0;
  private isOpen = false;
  private isClosed = false;
  private bufferTimer?: NodeJS.Timeout;
  private totalFrames = 0;
  private discardedFrames = 0;

  constructor(config: AudioGateConfig = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Push a PCM16 frame into the gate.
   * If gate is open, frame is immediately emitted.
   * If gate is closed, frame is buffered and checked for voice activity.
   */
  push(frame: Buffer): void {
    if (this.isClosed) return;
    this.totalFrames++;

    if (this.isOpen) {
      // Gate is open, pass through immediately
      this.emit('frame', frame);
      return;
    }

    // Gate is not open yet, buffer and analyze
    this.buffer.push(frame);
    
    const rms = this.calculateRMS(frame);
    if (rms >= this.config.rmsThreshold) {
      this.voicedCount++;
    }

    if (this.config.debug) {
      console.log(`[AudioGate] frame=${this.totalFrames} buffered=${this.buffer.length} voiced=${this.voicedCount} rms=${rms.toFixed(4)}`);
    }

    // Start timeout on first frame
    if (this.buffer.length === 1) {
      this.startBufferTimeout();
    }

    // Check if we should open the gate
    if (this.shouldOpen()) {
      this.openGate('voice_detected');
    } else if (this.buffer.length >= this.config.maxBufferFrames) {
      // Max buffer reached without voice detection - discard and reset
      this.resetBuffer('max_buffer_reached');
    }
  }

  /**
   * Signal that no more frames will arrive.
   * If gate never opened, buffer is discarded.
   * If gate is open, emits 'close'.
   */
  end(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    this.clearBufferTimeout();

    if (this.isOpen) {
      if (this.config.debug) {
        console.log(`[AudioGate] end() - gate was open, total_frames=${this.totalFrames}`);
      }
      this.emit('close');
    } else {
      // Gate never opened - discard buffer
      if (this.config.debug) {
        console.log(`[AudioGate] end() - gate never opened, discarding ${this.buffer.length} frames`);
      }
      this.discardedFrames += this.buffer.length;
      this.buffer = [];
      this.emit('discard', { frames: this.discardedFrames, reason: 'end_before_open' });
    }
  }

  /**
   * Force reset the gate state without emitting close.
   * Useful for cleanup without triggering downstream effects.
   */
  reset(): void {
    this.clearBufferTimeout();
    this.discardedFrames += this.buffer.length;
    this.buffer = [];
    this.voicedCount = 0;
    this.isOpen = false;
    this.isClosed = false;
    this.totalFrames = 0;
  }

  /** Returns true if the gate is currently open (passing frames through) */
  get opened(): boolean {
    return this.isOpen;
  }

  /** Returns true if end() has been called */
  get closed(): boolean {
    return this.isClosed;
  }

  /** Returns current buffer size */
  get bufferedFrames(): number {
    return this.buffer.length;
  }

  /** Returns stats for debugging */
  getStats(): { totalFrames: number; discardedFrames: number; bufferedFrames: number; voicedCount: number; isOpen: boolean } {
    return {
      totalFrames: this.totalFrames,
      discardedFrames: this.discardedFrames,
      bufferedFrames: this.buffer.length,
      voicedCount: this.voicedCount,
      isOpen: this.isOpen,
    };
  }

  private shouldOpen(): boolean {
    return (
      this.buffer.length >= this.config.minFrames &&
      this.voicedCount >= this.config.minVoicedFrames
    );
  }

  private openGate(reason: string): void {
    if (this.isOpen) return;
    this.isOpen = true;
    this.clearBufferTimeout();

    if (this.config.debug) {
      console.log(`[AudioGate] OPEN reason=${reason} buffered=${this.buffer.length} voiced=${this.voicedCount}`);
    }

    // Emit open event first
    this.emit('open');

    // Flush all buffered frames
    const buffered = this.buffer;
    this.buffer = [];
    for (const frame of buffered) {
      this.emit('frame', frame);
    }
  }

  private resetBuffer(reason: string): void {
    if (this.config.debug) {
      console.log(`[AudioGate] RESET reason=${reason} discarding=${this.buffer.length} frames`);
    }
    this.discardedFrames += this.buffer.length;
    this.buffer = [];
    this.voicedCount = 0;
    this.clearBufferTimeout();
    this.emit('discard', { frames: this.discardedFrames, reason });
  }

  private startBufferTimeout(): void {
    if (this.bufferTimer) return;
    this.bufferTimer = setTimeout(() => {
      this.bufferTimer = undefined;
      if (!this.isOpen && this.buffer.length > 0) {
        this.resetBuffer('timeout');
      }
    }, this.config.bufferTimeoutMs);
  }

  private clearBufferTimeout(): void {
    if (this.bufferTimer) {
      clearTimeout(this.bufferTimer);
      this.bufferTimer = undefined;
    }
  }

  /**
   * Calculate RMS (Root Mean Square) energy of a PCM16 frame.
   * Returns a value between 0 and 1.
   */
  private calculateRMS(frame: Buffer): number {
    // Ensure we're reading Int16 samples correctly
    const sampleCount = Math.floor(frame.length / 2);
    if (sampleCount === 0) return 0;

    let sumSquares = 0;
    for (let i = 0; i < sampleCount; i++) {
      // Read as signed 16-bit little-endian
      const sample = frame.readInt16LE(i * 2);
      const normalized = sample / 32768;
      sumSquares += normalized * normalized;
    }

    return Math.sqrt(sumSquares / sampleCount);
  }
}

export default AudioGate;
