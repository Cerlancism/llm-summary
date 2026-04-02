# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm test              # run all tests (vitest)
npm run test:watch    # watch mode
npx vitest run test/stream-guard.test.ts  # run a single test file
npm run typecheck     # type-check without emitting
npm run build         # compile TypeScript to dist/
npm start             # run CLI via tsx (dev mode)
```

## Architecture

Two-phase LLM summariser that fits text summaries within a target token range, using any OpenAI-compatible API.

**Phase 1 (Draft):** Calls the LLM with no length constraints to produce a quality-focused summary.

**Phase 2 (Fit):** Iteratively rewrites the draft to hit the target token range. Each attempt includes history of prior attempts so the model can calibrate. Temperature escalates on retries. Oldest history pairs are trimmed when `contextBudget` is exceeded.

### Key modules

- `src/summarizer.ts` — Core `summarise()` function implementing draft + fit phases. Uses Zod structured output (`zodResponseFormat`) to guarantee JSON-wrapped responses.
- `src/stream-guard.ts` — `streamWithRepetitionGuard()` wraps streaming completions with real-time repetition detection. On detecting degenerate loops, aborts and retries with escalated temperature/frequency/presence penalties.
- `src/openai-helpers.ts` — Low-level `streamChat()` that handles streaming, chunk callbacks (for abort signals), and graceful `length_limit` error handling.
- `src/index.ts` — Public API re-exports.
- `cli/index.ts` — CLI supporting file input, stdin pipe, and interactive REPL mode.

### Token counting

Uses `gpt-tokenizer` (BPE, cl100k_base) for token counting, not the API's token count. The fit phase trusts this local count for range checks.

### Tests

- `test/stream-guard.test.ts` — Unit tests for `detectRepetition` (pure, no API).
- `test/summarizer.test.ts` — Integration tests requiring a live OpenAI-compatible API (`OPENAI_API_KEY` in `.env`). Tests have long timeouts (up to 300s). Test outputs are written to `test/output/`.

## Conventions

- ESM-only (`"type": "module"` in package.json). Use `.js` extensions in imports even for `.ts` source files.
- Commit messages use conventional commits format (`feat:`, `fix:`, `docs:`, etc.).
