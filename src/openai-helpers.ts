import OpenAI from "openai";
import type { ChatCompletionStreamParams } from "openai/lib/ChatCompletionStream.js";

export type AbortReason = "repetition" | "length_limit" | null;

export interface StreamResult {
  content: string;
  aborted: boolean;
  abortReason: AbortReason;
  usage?: OpenAI.CompletionUsage;
}

export type { ChatCompletionStreamParams };

/**
 * Stream a chat completion, collecting content into a buffer.
 *
 * Accepts an optional `onChunk` callback for inspecting content as it arrives
 * (e.g. for repetition detection). If `onChunk` returns `true`, the stream
 * is aborted with reason "repetition".
 *
 * Handles the SDK's "length limit" error gracefully, returning the partial
 * buffer with `abortReason: "length_limit"`.
 *
 * Passes `stream_options: { include_usage: true }` so the final chunk
 * includes token usage stats.
 */
export async function streamChat(
  client: OpenAI,
  params: ChatCompletionStreamParams,
  onChunk?: (buffer: string, delta: string) => boolean | void
): Promise<StreamResult> {
  const stream = client.chat.completions.stream({
    ...params,
    stream_options: { include_usage: true },
  });

  let buffer = "";
  let aborted = false;
  let abortReason: AbortReason = null;

  try {
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? "";
      buffer += delta;

      if (onChunk) {
        const shouldAbort = onChunk(buffer, delta);
        if (shouldAbort) {
          aborted = true;
          abortReason = "repetition";
          stream.abort();
          break;
        }
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes("length limit")) {
      aborted = true;
      abortReason = "length_limit";
    } else {
      throw err;
    }
  }

  const finalMessage = aborted ? null : await stream.finalChatCompletion().catch(() => null);

  return {
    content: buffer,
    aborted,
    abortReason,
    usage: finalMessage?.usage ?? undefined,
  };
}
