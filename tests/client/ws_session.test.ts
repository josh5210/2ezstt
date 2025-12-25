import { WSSession, OutEvent } from '../../client/ws_session';
import WebSocket from 'ws';

// Mock WebSocket
jest.mock('ws');
const MockedWebSocket = WebSocket as jest.MockedClass<typeof WebSocket>;

describe('WSSession', () => {
  let mockWs: jest.Mocked<WebSocket>;
  let session: WSSession;
  let eventHandler: jest.MockedFunction<(ev: OutEvent) => void>;

  beforeEach(() => {
    mockWs = {
      on: jest.fn().mockReturnThis(),
      send: jest.fn(),
      close: jest.fn(),
      readyState: WebSocket.OPEN,
    } as any;

    MockedWebSocket.mockImplementation(() => mockWs);

    eventHandler = jest.fn();
    session = new WSSession('ws://test.com', {
      sessionId: 'test-session',
      speakerId: 'test-speaker',
      sampleRate: 16000
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
  });

  test('should send session.start on connect', async () => {
    // Set up the mock to trigger 'open' event
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      }
      return mockWs;
    });

    await session.connect(eventHandler);

    expect(MockedWebSocket).toHaveBeenCalledWith('ws://test.com', { perMessageDeflate: false });
    expect(mockWs.on).toHaveBeenCalledWith('open', expect.any(Function));
    expect(mockWs.on).toHaveBeenCalledWith('message', expect.any(Function));
    expect(mockWs.on).toHaveBeenCalledWith('close', expect.any(Function));
    expect(mockWs.on).toHaveBeenCalledWith('error', expect.any(Function));

    expect(mockWs.send).toHaveBeenCalledWith(JSON.stringify({
      type: 'session.start',
      session_id: 'test-session',
      speaker_id: 'test-speaker',
      sample_rate: 16000,
      format: 'pcm_s16le',
      transport: 'binary',
      meta: { app: 'ezstt-client' }
    }));
  });

  test('should handle incoming messages', async () => {
    let messageHandler: (data: Buffer) => void;

    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      } else if (event === 'message') {
        messageHandler = listener as (data: Buffer) => void;
      }
      return mockWs;
    });

    await session.connect(eventHandler);

    // Simulate incoming message
    const testEvent = { type: 'partial', text: 'hello world', utterance_id: 'test-id' };
    messageHandler!(Buffer.from(JSON.stringify(testEvent)));

    expect(eventHandler).toHaveBeenCalledWith(testEvent);
  });

  test('should handle malformed incoming messages gracefully', async () => {
    let messageHandler: (data: Buffer) => void;

    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      } else if (event === 'message') {
        messageHandler = listener as (data: Buffer) => void;
      }
      return mockWs;
    });

    await session.connect(eventHandler);

    // Send malformed JSON
    messageHandler!(Buffer.from('invalid json'));

    // Should not crash and should not call eventHandler
    expect(eventHandler).not.toHaveBeenCalled();
  });

  test('should send binary frames', async () => {
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      }
      return mockWs;
    });

    await session.connect(eventHandler);

    // Clear the session.start call
    mockWs.send.mockClear();

    const testFrame = Buffer.alloc(640, 42);
    session.writeFrame(testFrame);
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(mockWs.send).toHaveBeenCalledWith(testFrame, { binary: true });
  });

  test('should respect queue limits and drop oldest frames', async () => {
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      }
      return mockWs;
    });

    // Create session with smaller queue for testing
    const smallQueueSession = new (WSSession as any)('ws://test.com', {
      sessionId: 'test',
      speakerId: 'test'
    });
    smallQueueSession['maxQueue'] = 3; // Set small queue size for testing

    await smallQueueSession.connect(eventHandler);
    mockWs.send.mockClear();

    // Fill queue beyond capacity
    const frames = [
      Buffer.alloc(640, 1),
      Buffer.alloc(640, 2),
      Buffer.alloc(640, 3),
      Buffer.alloc(640, 4), // This should cause frame 1 to be dropped
    ];

    frames.forEach(frame => smallQueueSession.writeFrame(frame));
    await new Promise(resolve => setTimeout(resolve, 0));

    // Should have sent frames 2, 3, 4 (frame 1 dropped)
    expect(mockWs.send).toHaveBeenCalledTimes(3);
    expect(mockWs.send).not.toHaveBeenCalledWith(frames[0], { binary: true });
    expect(mockWs.send).toHaveBeenCalledWith(frames[1], { binary: true });
    expect(mockWs.send).toHaveBeenCalledWith(frames[2], { binary: true });
    expect(mockWs.send).toHaveBeenCalledWith(frames[3], { binary: true });
  });

  test('should send session.end on end()', async () => {
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any)(), 0);
      }
      return mockWs;
    });

    await session.connect(eventHandler);
    mockWs.send.mockClear();

    session.end();

    expect(mockWs.send).toHaveBeenCalledWith(JSON.stringify({ type: 'session.end' }));

    // Wait for the close timer to fire (with a reasonable timeout)
    await new Promise(resolve => setTimeout(resolve, 4100));

    expect(mockWs.close).toHaveBeenCalled();
  }, 6000);

  test('should not send frames when connection is closed', async () => {
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'open') {
        setTimeout(() => (listener as any).call(mockWs), 0);
      } else if (event === 'close') {
        // Simulate connection closing
        setTimeout(() => (listener as any).call(mockWs), 10);
      }
      return mockWs;
    });

    await session.connect(eventHandler);

    // Simulate connection close
    const closeHandler = mockWs.on.mock.calls.find(call => call[0] === 'close')?.[1] as ((...args:any[])=>void) | undefined;
    if (closeHandler) closeHandler.call(mockWs);

    mockWs.send.mockClear();

    const testFrame = Buffer.alloc(640, 42);
    session.writeFrame(testFrame);

    expect(mockWs.send).not.toHaveBeenCalled();
  });

  test('should reject connection promise on error', async () => {
    mockWs.on.mockImplementation((event: string | symbol, listener: (...args: any[]) => void) => {
      if (event === 'error') {
        setTimeout(() => (listener as any)(new Error('Connection failed')), 0);
      }
      return mockWs;
    });

    await expect(session.connect(eventHandler)).rejects.toThrow('Connection failed');
  });
});
