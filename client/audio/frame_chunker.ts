import { Transform, TransformCallback } from 'node:stream';

export class FrameChunker extends Transform {
  private carry: Buffer = Buffer.alloc(0);
  constructor(private readonly bytesPerFrame = 320 * 2) { super({ readableObjectMode: true }); }
  _transform(chunk: Buffer, _: BufferEncoding, cb: TransformCallback) {
    this.carry = Buffer.concat([this.carry, chunk]);
    while (this.carry.length >= this.bytesPerFrame) {
      const out = this.carry.subarray(0, this.bytesPerFrame);
      this.push(out); // push Buffer frame
      this.carry = this.carry.subarray(this.bytesPerFrame);
    }
    cb();
  }
  _flush(cb: TransformCallback) {
    // drop tail; endpointer server handles padding/silence
    this.carry = Buffer.alloc(0);
    cb();
  }
}