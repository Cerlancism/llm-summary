export { summarise, DEFAULT_MODEL, DEFAULT_MAX_FIT_ATTEMPTS, DEFAULT_CONTEXT_BUDGET } from "./summarizer.js";
export type { SummariseOptions, SummariseResult, TokenUsage } from "./summarizer.js";

export { streamChat } from "./openai-helpers.js";
export type { AbortReason, StreamResult, ChatCompletionStreamParams } from "./openai-helpers.js";

export { streamWithRepetitionGuard, detectRepetition } from "./stream-guard.js";
export type {
  RepetitionDetectorOptions,
  RetryEscalation,
  StreamGuardOptions,
  StreamGuardResult,
} from "./stream-guard.js";
