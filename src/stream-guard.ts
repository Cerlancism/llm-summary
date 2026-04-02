import OpenAI from "openai";
import {
  streamChat,
  type AbortReason,
  type ChatCompletionStreamParams,
  type StreamResult,
} from "./openai-helpers.js";

export type { AbortReason };

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

export interface StreamGuardResult extends StreamResult {
  /** The repeated pattern that triggered the abort, if any. */
  repetitionPattern?: string;
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

  const minLen = options.minPatternLength ?? 10;
  const maxLen = options.maxPatternLength ?? 1000;
  const minRepeats = options.minRepeats ?? 10;
  const warmup = options.warmupChars ?? 100;

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

    let repetitionPattern: string | undefined;

    const result = await streamChat(client, escalatedParams, (buffer) => {
      if (buffer.length < warmup) return;
      const pattern = detectRepetition(buffer, minLen, maxLen, minRepeats);
      if (pattern) {
        repetitionPattern = pattern;
        return true; // abort
      }
    });

    const guardResult: StreamGuardResult = {
      ...result,
      repetitionPattern,
      retries: retry,
    };

    if (!guardResult.aborted) {
      return guardResult;
    }

    if (verbose) {
      const reason = guardResult.abortReason === "length_limit"
        ? "length limit reached"
        : `repetition: "${repetitionPattern?.slice(0, 50)}…"`;
      console.log(`  ⚠ Aborted (retry ${retry}): ${reason}`);
    }

    // Length limit errors won't be fixed by escalating penalties — return
    // immediately and let the caller handle it (e.g. the fit loop retries).
    if (guardResult.abortReason === "length_limit") {
      return guardResult;
    }

    // Last retry still aborted — return what we have.
    if (retry === maxRetries) {
      return guardResult;
    }
  }

  // Unreachable, but TypeScript needs it.
  throw new Error("streamWithRepetitionGuard: unexpected exit");
}
