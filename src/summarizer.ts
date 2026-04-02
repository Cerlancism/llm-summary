import OpenAI from "openai";
import { encode } from "gpt-tokenizer";
import { streamWithRepetitionGuard } from "./stream-guard.js";

const MAX_FIT_ATTEMPTS = 5;

export interface SummariseOptions {
  model?: string;
  verbose?: boolean;
}

export interface SummariseResult {
  summary: string;
  tokens: number;
  attempts: number;
  withinRange: boolean;
}

function tokenCount(text: string): number {
  return encode(text).length;
}

// ─── Phase 1: Draft ──────────────────────────────────────────────────────────
// Generate a quality summary with no length pressure. The model focuses
// entirely on content — what to include, what to drop, how to structure it.

async function generateDraft(
  client: OpenAI,
  text: string,
  model: string,
  verbose: boolean
): Promise<{ draft: string; outputTokens: number | undefined; aborted: boolean }> {
  const result = await streamWithRepetitionGuard(client, {
    model,
    temperature: 0.3,
    messages: [
      {
        role: "system",
        content:
          "You are a summarisation assistant. Summarise the user's text accurately and completely. " +
          "Preserve all key facts. Output only the summary — no preamble or meta-commentary.",
      },
      { role: "user", content: `Summarise the following text:\n\n${text}` },
    ],
  });
  if (result.aborted && verbose) {
    console.log(`  ⚠ Draft aborted — repetition detected: "${result.repetitionPattern?.slice(0, 60)}…"`);
  }
  return {
    draft: result.content.trim(),
    outputTokens: result.usage?.completion_tokens,
    aborted: result.aborted,
  };
}

// ─── Phase 2: Fit ────────────────────────────────────────────────────────────
// Dedicated length-fitting pass. The model receives only the draft and
// length instructions — no original text, no distraction.
// Uses:
//   • max_tokens hard cap  → API prevents gross overshooting
//   • attempt history      → model can calibrate (like binary search)

async function fitLength(
  client: OpenAI,
  draft: string,
  minTokens: number,
  maxTokens: number,
  model: string,
  verbose: boolean
): Promise<{ summary: string; tokens: number; attempts: number }> {
  // Hard cap slightly above maxTokens to avoid mid-sentence truncation.
  const tokenCap = Math.ceil(maxTokens * 1.15);

  type Attempt = { tokens: number; text: string };
  const history: Attempt[] = [];

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "You are a length-fitting assistant. You receive a summary and must rewrite it " +
        "to fall within a strict token count range. Tokens are measured by a BPE tokenizer (GPT-style). " +
        "As a rough guide, 1 token ≈ 0.75 words or ~4 characters in English. " +
        "Output only the rewritten summary — no explanation, no token count, nothing else.",
    },
  ];

  let currentText = draft;
  let currentTokens = tokenCount(draft);

  for (let attempt = 1; attempt <= MAX_FIT_ATTEMPTS; attempt++) {
    const alreadyInRange = currentTokens >= minTokens && currentTokens <= maxTokens;

    if (alreadyInRange && attempt === 1) {
      if (verbose) console.log(`  Fit phase skipped — draft already in range (${currentTokens} tokens)`);
      return { summary: currentText, tokens: currentTokens, attempts: 0 };
    }

    const historyNote =
      history.length > 0
        ? `\nYour previous attempts and their token counts:\n` +
          history.map((h, i) => `  Attempt ${i + 1}: ${h.tokens} tokens`).join("\n") +
          "\nUse this to calibrate — aim for the middle of the target range.\n"
        : "";

    const direction = currentTokens < minTokens ? "too short" : "too long";
    const midTokens = Math.round((minTokens + maxTokens) / 2);
    const userMessage =
      attempt === 1
        ? `Rewrite this summary to be between ${minTokens} and ${maxTokens} tokens ` +
          `(aim for ~${midTokens} tokens).\n\nSummary:\n${currentText}`
        : `Still ${direction} (${currentTokens} tokens). Target: ${minTokens}–${maxTokens} tokens.\n` +
          historyNote +
          `Current summary:\n${currentText}`;

    messages.push({ role: "user", content: userMessage });

    const result = await streamWithRepetitionGuard(client, {
      model,
      temperature: 0.3,
      max_tokens: tokenCap,
      messages,
    });

    const fitted = result.content.trim();
    const fittedTokens = tokenCount(fitted);
    const apiTokens = result.usage?.completion_tokens;

    if (verbose) {
      console.log(
        `  Fit attempt ${attempt}: ${fittedTokens} tokens (gpt-tokenizer) / ${apiTokens ?? "?"} (api) | ` +
        `cap: ${tokenCap}${result.aborted ? ` | ⚠ aborted (repetition: "${result.repetitionPattern?.slice(0, 40)}…")` : ""}`
      );
    }

    history.push({ tokens: fittedTokens, text: fitted });
    messages.push({ role: "assistant", content: fitted });

    if (fittedTokens >= minTokens && fittedTokens <= maxTokens) {
      return { summary: fitted, tokens: fittedTokens, attempts: attempt };
    }

    currentText = fitted;
    currentTokens = fittedTokens;
  }

  // Return best attempt — closest to the midpoint of the target range.
  const mid = (minTokens + maxTokens) / 2;
  const best = history.reduce((a, b) =>
    Math.abs(a.tokens - mid) <= Math.abs(b.tokens - mid) ? a : b
  );
  return { summary: best.text, tokens: best.tokens, attempts: MAX_FIT_ATTEMPTS };
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

  if (verbose) console.log("  Phase 1: generating draft…");
  const { draft, outputTokens: draftApiTokens } = await generateDraft(client, text, model, verbose);
  const draftTokens = tokenCount(draft);
  if (verbose) {
    console.log(
      `  Draft: ${draftTokens} tokens (gpt-tokenizer) / ${draftApiTokens ?? "?"} (api)`
    );
  }

  if (verbose) console.log("  Phase 2: fitting to token range…");
  const { summary, tokens, attempts } = await fitLength(
    client,
    draft,
    minTokens,
    maxTokens,
    model,
    verbose
  );

  const withinRange = tokens >= minTokens && tokens <= maxTokens;
  return { summary, tokens, attempts: attempts + 1, withinRange };
}
