import { sendFinalSimpleMessage } from '../../client/finals_channel';

describe('sendFinalSimpleMessage', () => {
  test('sends timestamped message with final text', async () => {
    const send = jest.fn().mockResolvedValue(undefined);
    const now = () => new Date('2024-01-02T03:04:05.678Z');

    await sendFinalSimpleMessage({ send }, { text: 'Final result' }, 2000, now);

    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith('[2024-01-02T03:04:05.678Z] Final result');
  });

  test('skips empty text and respects message length limit', async () => {
    const send = jest.fn().mockResolvedValue(undefined);
    const now = () => new Date('2024-01-02T03:04:05.678Z');

    await sendFinalSimpleMessage({ send }, { text: '   ' }, 50, now);
    expect(send).not.toHaveBeenCalled();

    const longText = 'x'.repeat(60);
    await sendFinalSimpleMessage({ send }, { text: longText }, 20, now);

    expect(send).toHaveBeenCalledTimes(1);
    const expected = `[${now().toISOString()}] ${longText}`.slice(0, 20);
    expect(send).toHaveBeenCalledWith(expected);
  });
});
