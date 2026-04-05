import OpenAI from "openai";
import { encode } from "gpt-tokenizer";
import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";
import { streamWithRepetitionGuard } from "./stream-guard.js";

export interface SummariseOptions {
  model?: string;
  verbose?: boolean;
  /** Max fit attempts before returning best result. Default: 5 */
  maxFitAttempts?: number;
  /** Max tokens for the conversation context in the fit phase (excludes
   *  the system prompt, original text, and draft on the first attempt —
   *  only trims attempt history). Default: Infinity (no limit). */
  contextBudget?: number;
  /** Domain-specific instructions appended to the system prompt in both
   *  draft and fit phases (e.g. language, style, or format constraints). */
  instructions?: string;
}

export interface TokenUsage {
  input: number;
  output: number;
  total: number;
}

export interface SummariseResult {
  summary: string;
  tokens: number;
  attempts: number;
  withinRange: boolean;
  usage: TokenUsage;
  durationMs: number;
  outputTokensPerSecond: number;
}

function tokenCount(text: string): number {
  return encode(text).length;
}

// ─── Zod schemas ─────────────────────────────────────────────────────────────

const DraftResponse = z.object({
  summary: z.string().describe("The summarised text"),
});

const FitResponse = z.object({
  summary: z.string().describe("The rewritten summary fitted to the target token range"),
});

function parseSummary(json: string): string {
  try {
    const parsed = JSON.parse(json);
    return (parsed.summary ?? "").trim();
  } catch {
    // Fallback: if JSON parsing fails (e.g. aborted stream), return raw content
    return json.trim();
  }
}

// ─── Phase 1: Draft ──────────────────────────────────────────────────────────
// Generate a quality summary with no length pressure. The model focuses
// entirely on content — what to include, what to drop, how to structure it.

async function generateDraft(
  client: OpenAI,
  text: string,
  model: string,
  verbose: boolean,
  instructions?: string
): Promise<{ draft: string; promptTokens: number; completionTokens: number; aborted: boolean }> {
  let systemContent =
    "You are a summarisation assistant. Summarise the user's text accurately and completely. " +
    "Preserve all key facts.";
  if (instructions) {
    systemContent += `\n\nAdditional instructions:\n${instructions}`;
  }

  const result = await streamWithRepetitionGuard(client, {
    model,
    temperature: 0,
    messages: [
      {
        role: "system",
        content: systemContent,
      },
      { role: "user", content: `Summarise the following text:\n\n${text}` },
    ],
    response_format: zodResponseFormat(DraftResponse, "draft_response"),
  }, { verbose });

  const summary = parseSummary(result.content);
  const promptTokens = result.usage?.prompt_tokens ?? 0;
  const completionTokens = result.usage?.completion_tokens ?? 0;

  if (verbose) {
    console.log(`    usage: ${promptTokens} prompt + ${completionTokens} completion = ${promptTokens + completionTokens} total`);
  }

  return { draft: summary, promptTokens, completionTokens, aborted: result.aborted };
}

// ─── Phase 2: Fit ────────────────────────────────────────────────────────────
// Dedicated length-fitting pass. The model receives the original text, the
// draft, and length instructions. Previous attempts and their token counts
// are included so the model can calibrate.
// Uses:
//   • max_tokens hard cap  → API prevents gross overshooting
//   • attempt history      → model can calibrate (like binary search)
//   • original text        → context for quality rewrites

async function fitLength(
  client: OpenAI,
  originalText: string,
  draft: string,
  minTokens: number,
  maxTokens: number,
  opts: Required<SummariseOptions>
): Promise<{ summary: string; tokens: number; attempts: number; inputTokens: number; outputTokens: number }> {
  const { model, verbose, maxFitAttempts: maxAttempts, contextBudget, instructions } = opts;
  // Hard cap above maxTokens: 2x buffer for JSON wrapper overhead + model variance.
  const maxTokenTarget = maxTokens * 2;
  const tokenCap = contextBudget < Infinity
    ? Math.min(maxTokenTarget, contextBudget)
    : maxTokenTarget;

  type Attempt = { tokens: number; text: string; userMsg: string; assistantMsg: string };
  const history: Attempt[] = [];
  let inputTokens = 0;
  let outputTokens = 0;

  let systemContent =
    "You are a length-fitting assistant. You receive an original text, a summary draft, and " +
    "must rewrite the summary to fall within a strict token count range. Tokens are measured " +
    "by a BPE tokenizer (GPT-style). As a rough guide, 1 token ≈ 0.75 words or ~4 characters " +
    "in English. You may refer to the original text for context but your output must be a " +
    "standalone summary within the target token range.";
  if (instructions) {
    systemContent += `\n\nAdditional instructions (preserve these requirements when rewriting):\n${instructions}`;
  }

  const systemMessage: OpenAI.Chat.ChatCompletionMessageParam = {
    role: "system",
    content: systemContent,
  };

  let currentText = draft;
  let currentTokens = tokenCount(draft);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const alreadyInRange = currentTokens >= minTokens && currentTokens <= maxTokens;

    if (alreadyInRange && attempt === 1) {
      if (verbose) console.log(`  Fit phase skipped — draft already in range (${currentTokens} tokens)`);
      return { summary: currentText, tokens: currentTokens, attempts: 0, inputTokens: 0, outputTokens: 0 };
    }

    const tooShort = currentTokens < minTokens;
    const delta = tooShort ? minTokens - currentTokens : currentTokens - maxTokens;
    const midTokens = Math.round((minTokens + maxTokens) / 2);
    const direction = tooShort ? "too short" : "too long";
    const guidance = tooShort
      ? `It is ${delta} tokens below the minimum (${minTokens}). Add more detail, examples, or supporting points to expand it.`
      : `It is ${delta} tokens above the maximum (${maxTokens}). Remove less important details, combine sentences, or use more concise phrasing to shorten it.`;

    const userContent =
      attempt === 1
        ? `Rewrite this summary to be between ${minTokens} and ${maxTokens} tokens ` +
          `(aim for ~${midTokens} tokens). ` +
          `The current draft is ${currentTokens} tokens — ${direction}. ${guidance}\n\n` +
          `Original text:\n${originalText}\n\n` +
          `Draft summary:\n${currentText}`
        : `Still ${direction} (${currentTokens} tokens). Target: ${minTokens}–${maxTokens} tokens. ${guidance}\n` +
          `Current summary:\n${currentText}`;

    // Build messages: system + trimmed history + current user message.
    // History is trimmed from the front (oldest first) to stay within contextBudget.
    let historyMessages: OpenAI.Chat.ChatCompletionMessageParam[] = history.flatMap(h => [
      { role: "user" as const, content: h.userMsg },
      { role: "assistant" as const, content: h.assistantMsg },
    ]);

    if (contextBudget < Infinity && historyMessages.length > 0) {
      let historyTokens = historyMessages.reduce((sum, m) =>
        sum + tokenCount(typeof m.content === "string" ? m.content : ""), 0);
      // Drop oldest pairs (user+assistant) until within budget
      while (historyTokens > contextBudget && historyMessages.length >= 2) {
        const dropped1 = historyMessages.shift()!;
        const dropped2 = historyMessages.shift()!;
        historyTokens -= tokenCount(typeof dropped1.content === "string" ? dropped1.content : "");
        historyTokens -= tokenCount(typeof dropped2.content === "string" ? dropped2.content : "");
        if (verbose) console.log(`    (trimmed oldest history pair to fit context budget)`);
      }
    }

    // Also inject a history note summarising previous attempt token counts
    const historyNote =
      history.length > 0
        ? `\nYour previous attempts and their token counts:\n` +
          history.map((h, i) => `  Attempt ${i + 1}: ${h.tokens} tokens`).join("\n") +
          "\nUse this to calibrate — aim for the middle of the target range.\n"
        : "";
    const fullUserContent = attempt > 1 && historyNote
      ? userContent.replace(`${guidance}\n`, `${guidance}\n${historyNote}`)
      : userContent;

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      systemMessage,
      ...historyMessages,
      { role: "user", content: fullUserContent },
    ];

    // Escalate temperature on subsequent attempts to encourage variation.
    const temperature = Math.min((attempt - 1) * 0.2, 1.2);

    const result = await streamWithRepetitionGuard(client, {
      model,
      temperature,
      max_tokens: tokenCap,
      messages,
      response_format: zodResponseFormat(FitResponse, "fit_response"),
    }, { verbose });

    const fitted = parseSummary(result.content);
    const fittedTokens = tokenCount(fitted);
    const promptToks = result.usage?.prompt_tokens ?? 0;
    const completionToks = result.usage?.completion_tokens ?? 0;
    const attemptTotal = promptToks + completionToks;
    inputTokens += promptToks;
    outputTokens += completionToks;

    if (verbose) {
      console.log(
        `  Fit attempt ${attempt}: ${fittedTokens} tokens (gpt-tokenizer) / ${completionToks} (api) | ` +
        `temp: ${temperature} | cap: ${tokenCap}${result.aborted ? ` | ⚠ ${result.abortReason} after ${result.retries} retries` : ""}`
      );
      console.log(`    usage: ${promptToks} prompt + ${completionToks} completion = ${attemptTotal} total`);
      console.log(`    text: ${fitted}`);
    }

    history.push({ tokens: fittedTokens, text: fitted, userMsg: fullUserContent, assistantMsg: result.content });

    if (fittedTokens >= minTokens && fittedTokens <= maxTokens) {
      return { summary: fitted, tokens: fittedTokens, attempts: attempt, inputTokens, outputTokens };
    }

    currentText = fitted;
    currentTokens = fittedTokens;
  }

  // Return best attempt — closest to the midpoint of the target range.
  const mid = (minTokens + maxTokens) / 2;
  const best = history.reduce((a, b) =>
    Math.abs(a.tokens - mid) <= Math.abs(b.tokens - mid) ? a : b
  );
  return { summary: best.text, tokens: best.tokens, attempts: maxAttempts, inputTokens, outputTokens };
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Two-phase summarisation:
 *   1. Draft  — generate a quality summary with no length pressure
 *   2. Fit    — dedicated length-fitting pass with max_tokens cap + history
 */
export async function summarise(
  client: OpenAI,
  text: string,
  minTokens: number,
  maxTokens: number,
  options: SummariseOptions = {}
): Promise<SummariseResult> {
  if (minTokens > maxTokens) throw new Error("minTokens must be <= maxTokens");
  if (minTokens < 1) throw new Error("minTokens must be >= 1");

  const model = options.model ?? process.env.OPENAI_MODEL ?? "gpt-4o-mini";
  const verbose = options.verbose ?? false;

  const startTime = Date.now();

  if (verbose) console.log("  Phase 1: generating draft…");
  const { draft, promptTokens: draftPrompt, completionTokens: draftCompletion } =
    await generateDraft(client, text, model, verbose, options.instructions);
  const draftTokens = tokenCount(draft);
  if (verbose) {
    console.log(`  Draft: ${draftTokens} tokens (gpt-tokenizer)`);
    console.log(`    text: ${draft}`);
  }

  const maxFitAttempts = options.maxFitAttempts ?? 5;
  const contextBudget = options.contextBudget ?? Infinity;

  if (verbose) console.log("  Phase 2: fitting to token range…");
  const { summary, tokens, attempts, inputTokens: fitInput, outputTokens: fitOutput } = await fitLength(
    client,
    text,
    draft,
    minTokens,
    maxTokens,
    { model, verbose, maxFitAttempts, contextBudget, instructions: options.instructions ?? "" }
  );

  const usage: TokenUsage = {
    input: draftPrompt + fitInput,
    output: draftCompletion + fitOutput,
    total: draftPrompt + draftCompletion + fitInput + fitOutput,
  };
  const durationMs = Date.now() - startTime;
  const outputTokensPerSecond = durationMs > 0 ? (usage.output / durationMs) * 1000 : 0;

  if (verbose) {
    console.log(`  Total usage: ${usage.input} input + ${usage.output} output = ${usage.total} total`);
    console.log(`  Duration: ${(durationMs / 1000).toFixed(1)}s | Output rate: ${outputTokensPerSecond.toFixed(1)} tokens/s`);
  }

  const withinRange = tokens >= minTokens && tokens <= maxTokens;
  return { summary, tokens, attempts: attempts + 1, withinRange, usage, durationMs, outputTokensPerSecond };
}
