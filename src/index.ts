/**
 * CLI entry point.
 *
 * Usage:
 *   npx tsx src/index.ts <minTokens> <maxTokens> [textFile]
 *
 * If textFile is omitted, reads from stdin.
 *
 * Examples:
 *   echo "Long article..." | npx tsx src/index.ts 50 80
 *   npx tsx src/index.ts 100 150 article.txt
 */

import "dotenv/config";
import fs from "node:fs";
import readline from "node:readline";
import OpenAI from "openai";
import { summarise } from "./summarizer.js";

async function readStdin(): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin });
  const lines: string[] = [];
  for await (const line of rl) lines.push(line);
  return lines.join("\n");
}

async function main(): Promise<void> {
  const [, , minArg, maxArg, fileArg] = process.argv;

  if (!minArg || !maxArg) {
    console.error("Usage: npx tsx src/index.ts <minTokens> <maxTokens> [textFile]");
    process.exit(1);
  }

  const minTokens = parseInt(minArg, 10);
  const maxTokens = parseInt(maxArg, 10);

  if (isNaN(minTokens) || isNaN(maxTokens)) {
    console.error("minTokens and maxTokens must be integers.");
    process.exit(1);
  }

  const text = fileArg ? fs.readFileSync(fileArg, "utf8") : await readStdin();

  if (!text.trim()) {
    console.error("No input text provided.");
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

  console.log(`Summarising to ${minTokens}–${maxTokens} tokens…\n`);

  const result = await summarise(client, text, minTokens, maxTokens, { verbose: true });

  console.log("\n--- Summary ---");
  console.log(result.summary);
  console.log("\n--- Stats ---");
  console.log(`Tokens  : ${result.tokens}`);
  console.log(`Range   : ${minTokens}–${maxTokens}`);
  console.log(`Status  : ${result.withinRange ? "✓ within range" : "✗ out of range after max attempts"}`);
  console.log(`Attempts: ${result.attempts}`);
}

main().catch((err: Error) => {
  console.error(err.message);
  process.exit(1);
});
