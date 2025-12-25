export interface FinalsChannelLike {
  send(content: string): Promise<unknown>;
}

export interface FinalLikeEvent {
  text?: string | null;
}

export async function sendFinalSimpleMessage(
  channel: FinalsChannelLike | null | undefined,
  ev: FinalLikeEvent,
  limit = 2000,
  now: () => Date = () => new Date(),
): Promise<void> {
  if (!channel) return;
  const text = typeof ev.text === 'string' ? ev.text.trim() : '';
  if (!text) return;
  const timestamp = now().toISOString();
  let content = `[${timestamp}] ${text}`;
  if (content.length > limit) {
    content = content.slice(0, limit);
  }
  await channel.send(content);
}
