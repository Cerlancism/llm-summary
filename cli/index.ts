#!/usr/bin/env node

/**
 * CLI entry point.
 *
 * Usage:
 *   npx tsx cli/index.ts <minTokens> <maxTokens> [textFile]
 *
 * If textFile is omitted and stdin is piped, reads from stdin.
 * If stdin is a TTY, enters interactive mode: paste text, press
 * Ctrl+D (EOF) to summarise, then repeat. Ctrl+C to exit.
 *
 * Examples:
 *   echo "Long article..." | npx tsx cli/index.ts 50 80
 *   npx tsx cli/index.ts 100 150 article.txt
 *   npx tsx cli/index.ts 50 80          # interactive mode
 */

import "dotenv/config";
import fs from "node:fs";
import readline from "node:readline";
import OpenAI from "openai";
import { summarise } from "../src/summarizer.js";

async function readStdin(): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin });
  const lines: string[] = [];
  for await (const line of rl) lines.push(line);
  return lines.join("\n");
}

function printResult(
  result: Awaited<ReturnType<typeof summarise>>,
  minTokens: number,
  maxTokens: number
): void {
  console.log("\n--- Summary ---");
  console.log(result.summary);
  console.log("\n--- Stats ---");
  console.log(`Tokens  : ${result.tokens}`);
  console.log(`Range   : ${minTokens}–${maxTokens}`);
  console.log(`Status  : ${result.withinRange ? "✓ within range" : "✗ out of range after max attempts"}`);
  console.log(`Attempts: ${result.attempts}`);
  console.log(`Usage   : ${result.usage.input} input + ${result.usage.output} output = ${result.usage.total} total`);
}

async function runOnce(
  client: OpenAI,
  text: string,
  minTokens: number,
  maxTokens: number
): Promise<void> {
  const model = process.env.OPENAI_MODEL;
  console.log(`Summarising to ${minTokens}–${maxTokens} tokens…\n`);
  const result = await summarise(client, text, minTokens, maxTokens, { ...(model && { model }), verbose: true });
  printResult(result, minTokens, maxTokens);
}

async function repl(
  client: OpenAI,
  minTokens: number,
  maxTokens: number
): Promise<void> {
  console.log(`Interactive mode (${minTokens}–${maxTokens} tokens). Paste text then press Ctrl+D to summarise. Ctrl+C to exit.\n`);

  while (true) {
    process.stdout.write("> ");

    const lines: string[] = [];
    const rl = readline.createInterface({ input: process.stdin });
    for await (const line of rl) lines.push(line);
    const text = lines.join("\n").trim();

    if (!text) {
      console.log("(empty input, exiting)");
      break;
    }

    await runOnce(client, text, minTokens, maxTokens);
    console.log();

    // stdin closed (Ctrl+D sends EOF which closes the stream)
    // Re-open only works if the terminal is still alive
    if (!process.stdin.readable) break;
  }
}

async function main(): Promise<void> {
  const [, , minArg, maxArg, fileArg] = process.argv;

  if (!minArg || !maxArg) {
    console.error("Usage: npx tsx cli/index.ts <minTokens> <maxTokens> [textFile]");
    process.exit(1);
  }

  const minTokens = parseInt(minArg, 10);
  const maxTokens = parseInt(maxArg, 10);

  if (isNaN(minTokens) || isNaN(maxTokens)) {
    console.error("minTokens and maxTokens must be integers.");
    process.exit(1);
  }

  if (!process.env.OPENAI_API_KEY) {
    console.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.");
    process.exit(1);
  }

  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });

  if (fileArg) {
    const text = fs.readFileSync(fileArg, "utf8");
    if (!text.trim()) {
      console.error("File is empty.");
      process.exit(1);
    }
    await runOnce(client, text, minTokens, maxTokens);
  } else if (!process.stdin.isTTY) {
    // Piped input
    const text = await readStdin();
    if (!text.trim()) {
      console.error("No input text provided.");
      process.exit(1);
    }
    await runOnce(client, text, minTokens, maxTokens);
  } else {
    // Interactive TTY mode
    await repl(client, minTokens, maxTokens);
  }
}

main().catch((err: Error) => {
  console.error(err.message);
  process.exit(1);
});
