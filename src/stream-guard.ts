import OpenAI from "openai";
import type { ChatCompletionStreamParams } from "openai/lib/ChatCompletionStream.js";

export interface RepetitionDetectorOptions {
  /** Shortest pattern length to watch for (chars). Default: 10 */
  minPatternLength?: number;
  /** Longest pattern length to watch for (chars). Default: 1000 */
  maxPatternLength?: number;
  /** How many consecutive repeats triggers abort. Default: 10 */
  minRepeats?: number;
  /** Only run detection after accumulating this many chars. Default: 100 */
  warmupChars?: number;
}

export interface RetryEscalation {
  /** Temperature increment per retry. Default: 0.2 */
  temperatureStep?: number;
  /** Frequency penalty increment per retry. Default: 0.3 */
  frequencyPenaltyStep?: number;
  /** Presence penalty increment per retry. Default: 0.2 */
  presencePenaltyStep?: number;
}

export interface StreamGuardOptions extends RepetitionDetectorOptions {
  /** Max retries on repetition abort. Default: 3 */
  maxRetries?: number;
  /** Per-retry parameter escalation. */
  escalation?: RetryEscalation;
  /** Log retry info. Default: false */
  verbose?: boolean;
}

export interface StreamGuardResult {
  content: string;
  aborted: boolean;
  /** The repeated pattern that triggered the abort, if any. */
  repetitionPattern?: string;
  usage?: OpenAI.CompletionUsage;
  /** How many retries were attempted (0 = first try succeeded). */
  retries: number;
}

/**
 * Scan the tail of `buffer` for a repeated pattern.
 */
export function detectRepetition(
  buffer: string,
  minLen: number,
  maxLen: number,
  minRepeats: number
): string | null {
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
 * Run a single streaming attempt, aborting on repetition.
 */
async function runStream(
  client: OpenAI,
  params: ChatCompletionStreamParams,
  detectorOpts: Required<RepetitionDetectorOptions>
): Promise<StreamGuardResult> {
  const { minPatternLength, maxPatternLength, minRepeats, warmupChars } = detectorOpts;

  const stream = client.chat.completions.stream({ ...params });

  let buffer = "";
  let aborted = false;
  let repetitionPattern: string | undefined;

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    buffer += delta;

    if (buffer.length < warmupChars) continue;

    const pattern = detectRepetition(buffer, minPatternLength, maxPatternLength, minRepeats);
    if (pattern) {
      repetitionPattern = pattern;
      aborted = true;
      stream.abort();
      break;
    }
  }

  const finalMessage = aborted ? null : await stream.finalChatCompletion().catch(() => null);

  return {
    content: buffer,
    aborted,
    repetitionPattern,
    usage: finalMessage?.usage ?? undefined,
    retries: 0,
  };
}

/**
 * Stream a chat completion with repetition detection and automatic retry.
 *
 * On each retry, escalates temperature, frequency_penalty, and presence_penalty
 * to push the model out of degenerate loops.
 */
export async function streamWithRepetitionGuard(
  client: OpenAI,
  params: ChatCompletionStreamParams,
  options: StreamGuardOptions = {}
): Promise<StreamGuardResult> {
  const maxRetries = options.maxRetries ?? 3;
  const verbose = options.verbose ?? false;
  const tempStep = options.escalation?.temperatureStep ?? 0.2;
  const freqStep = options.escalation?.frequencyPenaltyStep ?? 0.3;
  const presStep = options.escalation?.presencePenaltyStep ?? 0.2;

  const detector: Required<RepetitionDetectorOptions> = {
    minPatternLength: options.minPatternLength ?? 10,
    maxPatternLength: options.maxPatternLength ?? 1000,
    minRepeats: options.minRepeats ?? 10,
    warmupChars: options.warmupChars ?? 100,
  };

  const baseTemp = (params.temperature as number | undefined) ?? 0;
  const baseFreq = (params.frequency_penalty as number | undefined) ?? 0;
  const basePres = (params.presence_penalty as number | undefined) ?? 0;

  for (let retry = 0; retry <= maxRetries; retry++) {
    const escalatedParams: ChatCompletionStreamParams = {
      ...params,
      temperature: Math.min(baseTemp + tempStep * retry, 2),
      frequency_penalty: Math.min(baseFreq + freqStep * retry, 2),
      presence_penalty: Math.min(basePres + presStep * retry, 2),
    };

    if (verbose && retry > 0) {
      console.log(
        `  Stream retry ${retry}/${maxRetries} — temp: ${escalatedParams.temperature}, ` +
        `freq_penalty: ${escalatedParams.frequency_penalty}, pres_penalty: ${escalatedParams.presence_penalty}`
      );
    }

    const result = await runStream(client, escalatedParams, detector);

    if (!result.aborted) {
      return { ...result, retries: retry };
    }

    if (verbose) {
      console.log(`  ⚠ Repetition detected (retry ${retry}): "${result.repetitionPattern?.slice(0, 50)}…"`);
    }

    // Last retry still aborted — return what we have.
    if (retry === maxRetries) {
      return { ...result, retries: retry };
    }
  }

  // Unreachable, but TypeScript needs it.
  throw new Error("streamWithRepetitionGuard: unexpected exit");
}
