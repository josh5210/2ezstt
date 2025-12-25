import { FrameChunker } from '../../../client/audio/frame_chunker';

describe('FrameChunker', () => {
  const FRAME_SIZE = 320 * 2; // 640 bytes for 20ms at 16kHz mono

  test('should output exact frame sizes', (done) => {
    const chunker = new FrameChunker();
    const frames: Buffer[] = [];

    chunker.on('data', (frame: Buffer) => {
      expect(frame.length).toBe(FRAME_SIZE);
      frames.push(frame);
    });

    chunker.on('end', () => {
      expect(frames.length).toBe(3); // Should get exactly 3 complete frames
      done();
    });

    // Write exactly 3 frames worth of data in various chunk sizes
    const totalData = Buffer.alloc(FRAME_SIZE * 3);
    for (let i = 0; i < totalData.length; i++) {
      totalData[i] = i % 256; // Fill with test pattern
    }

    // Send in irregular chunks
    chunker.write(totalData.subarray(0, 100));
    chunker.write(totalData.subarray(100, 800));
    chunker.write(totalData.subarray(800, 1500));
    chunker.write(totalData.subarray(1500));
    chunker.end();
  });

  test('should handle varying input chunk sizes', (done) => {
    const chunker = new FrameChunker();
    const frames: Buffer[] = [];

    chunker.on('data', (frame: Buffer) => {
      expect(frame.length).toBe(FRAME_SIZE);
      frames.push(frame);
    });

    chunker.on('end', () => {
      expect(frames.length).toBe(2);

      // Verify content integrity
      const reconstructed = Buffer.concat(frames);
      expect(reconstructed.length).toBe(FRAME_SIZE * 2);

      // Check that data was preserved correctly
      for (let i = 0; i < reconstructed.length; i++) {
        expect(reconstructed[i]).toBe(i % 256);
      }

      done();
    });

    // Send 2.5 frames worth of data in small chunks
    const totalData = Buffer.alloc(Math.floor(FRAME_SIZE * 2.5));
    for (let i = 0; i < totalData.length; i++) {
      totalData[i] = i % 256;
    }

    // Send in very small chunks to test buffering
    const chunkSize = 50;
    for (let i = 0; i < totalData.length; i += chunkSize) {
      const end = Math.min(i + chunkSize, totalData.length);
      chunker.write(totalData.subarray(i, end));
    }
    chunker.end();
  });

  test('should drop incomplete tail on flush', (done) => {
    const chunker = new FrameChunker();
    const frames: Buffer[] = [];

    chunker.on('data', (frame: Buffer) => {
      frames.push(frame);
    });

    chunker.on('end', () => {
      expect(frames.length).toBe(1); // Only one complete frame
      expect(frames[0].length).toBe(FRAME_SIZE);
      done();
    });

    // Send 1.7 frames worth of data
    const incompleteData = Buffer.alloc(Math.floor(FRAME_SIZE * 1.7));
    chunker.write(incompleteData);
    chunker.end(); // Should drop the partial frame
  });

  test('should handle exact frame boundaries', (done) => {
    const chunker = new FrameChunker();
    const frames: Buffer[] = [];

    chunker.on('data', (frame: Buffer) => {
      frames.push(frame);
    });

    chunker.on('end', () => {
      expect(frames.length).toBe(5);
      frames.forEach(frame => {
        expect(frame.length).toBe(FRAME_SIZE);
      });
      done();
    });

    // Send exactly 5 frames, one at a time
    for (let i = 0; i < 5; i++) {
      const frame = Buffer.alloc(FRAME_SIZE, i); // Fill with frame number
      chunker.write(frame);
    }
    chunker.end();
  });

  test('should handle empty input', (done) => {
    const chunker = new FrameChunker();
    const frames: Buffer[] = [];

    chunker.on('data', (frame: Buffer) => {
      frames.push(frame);
    });

    chunker.on('end', () => {
      expect(frames.length).toBe(0);
      done();
    });

    chunker.end(); // End immediately without writing anything
  });
});