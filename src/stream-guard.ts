import OpenAI from "openai";
import type { ChatCompletionStreamParams } from "openai/lib/ChatCompletionStream.js";

export interface RepetitionDetectorOptions {
  /** Shortest pattern length to watch for (chars). Default: 5 */
  minPatternLength?: number;
  /** Longest pattern length to watch for (chars). Default: 300 */
  maxPatternLength?: number;
  /** How many consecutive repeats triggers abort. Default: 3 */
  minRepeats?: number;
  /** Only run detection after accumulating this many chars. Default: 50 */
  warmupChars?: number;
}

export interface StreamGuardResult {
  content: string;
  aborted: boolean;
  /** The repeated pattern that triggered the abort, if any. */
  repetitionPattern?: string;
  usage?: OpenAI.CompletionUsage;
}

/**
 * Scan the tail of `buffer` for a repeated pattern.
 *
 * Checks every candidate length from minLen to maxLen.
 * For each, takes the last `len` chars as the candidate pattern
 * and verifies that the `minRepeats - 1` blocks immediately
 * preceding it are identical.
 *
 * Only inspects the last (maxLen * minRepeats) chars — O(maxLen²) per call
 * but with small constants and early-exit on first match.
 */
function detectRepetition(
  buffer: string,
  minLen: number,
  maxLen: number,
  minRepeats: number
): string | null {
  // Limit the search window to avoid scanning the whole buffer each time.
  const tail = buffer.slice(-(maxLen * minRepeats));
  const tailLen = tail.length;

  for (let len = minLen; len <= Math.min(maxLen, Math.floor(tailLen / minRepeats)); len++) {
    const candidate = tail.slice(tailLen - len);

    let allMatch = true;
    for (let r = 1; r < minRepeats; r++) {
      const start = tailLen - len * (r + 1);
      const end = tailLen - len * r;
      if (start < 0 || tail.slice(start, end) !== candidate) {
        allMatch = false;
        break;
      }
    }

    if (allMatch) return candidate;
  }

  return null;
}

/**
 * Stream a chat completion and abort early if a repetition loop is detected.
 *
 * @example
 * const result = await streamWithRepetitionGuard(client, {
 *   model: "gpt-4o-mini",
 *   messages: [{ role: "user", content: "..." }],
 * });
 * if (result.aborted) {
 *   console.warn("Aborted — repetition detected:", result.repetitionPattern);
 * }
 */
export async function streamWithRepetitionGuard(
  client: OpenAI,
  params: ChatCompletionStreamParams,
  options: RepetitionDetectorOptions = {}
): Promise<StreamGuardResult> {
  const minLen = options.minPatternLength ?? 10;
  const maxLen = options.maxPatternLength ?? 1000;
  const minRepeats = options.minRepeats ?? 10;
  const warmup = options.warmupChars ?? 100;

  const stream = client.chat.completions.stream({
    ...params,
  });

  let buffer = "";
  let aborted = false;
  let repetitionPattern: string | undefined;

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    buffer += delta;

    // Skip detection until we have enough content to be meaningful.
    if (buffer.length < warmup) continue;

    const pattern = detectRepetition(buffer, minLen, maxLen, minRepeats);
    if (pattern) {
      repetitionPattern = pattern;
      aborted = true;
      stream.abort();
      break;
    }
  }

  // Collect final usage if the stream completed normally.
  const finalMessage = aborted ? null : await stream.finalChatCompletion().catch(() => null);

  return {
    content: buffer,
    aborted,
    repetitionPattern,
    usage: finalMessage?.usage ?? undefined,
  };
}
