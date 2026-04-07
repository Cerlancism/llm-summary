# llm-summary

LLM-powered text summarisation that reliably fits within a target token range.

## How it works

The summariser uses a two-phase approach:

1. **Draft** -- Generate a quality summary with no length pressure, letting the model focus entirely on content.
2. **Fit** -- Iteratively rewrite the summary to land within the target token range, using attempt history and escalating temperature to converge.

A streaming repetition guard detects degenerate loops during generation and automatically retries with escalated parameters (temperature, frequency/presence penalty).

## Install from git

```bash
npm install git+https://github.com/Cerlancism/llm-summary.git
```

## Environment File for CLI

Copy `.env.example` to `.env` and configure your API key:

```bash
cp .env.example .env
```

Works with OpenAI-compatible APIs (OpenAI, Ollama, etc). Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and optionally `OPENAI_MODEL` in `.env`.

## CLI usage

```bash
# From a file
npx tsx cli/index.ts 50 80 article.txt

# From stdin
echo "Long article..." | npx tsx cli/index.ts 50 80

# Interactive mode (paste text, Ctrl+D to summarise, Ctrl+C to exit)
npx tsx cli/index.ts 50 80
```

After building (`npm run build`), the CLI is also available as:

```bash
llm-summary 50 80 article.txt
```

## Library usage

```typescript
import OpenAI from "openai";
import { summarise } from "llm-summary";

const client = new OpenAI();
const result = await summarise(client, text, 50, 80, {
  model: "gpt-4o-mini",  // optional, defaults to OPENAI_MODEL or gpt-4o-mini
  verbose: true,          // optional, logs progress
  maxFitAttempts: 5,      // optional, max fit iterations
  contextBudget: 4000,    // optional, max tokens for fit conversation history
  instructions: "Write in Traditional Chinese. Preserve proper nouns in their original language.",
                          // optional, domain-specific instructions for both phases
});

console.log(result.summary);    // the summary text
console.log(result.tokens);     // token count (gpt-tokenizer)
console.log(result.withinRange); // true if within [min, max]
console.log(result.attempts);   // total LLM calls
console.log(result.usage);      // { input, output, total } token usage
```

### Exports

- `summarise` -- Main two-phase summarisation function
- `streamChat` -- Low-level streaming chat completion helper
- `streamWithRepetitionGuard` -- Streaming with automatic repetition detection and retry
- `detectRepetition` -- Pattern repetition detector (useful standalone)


## Development

```bash
npm test           # run tests (vitest)
npm run test:watch # watch mode
npm run typecheck  # type-check without emitting
npm run build      # compile to dist/
```
