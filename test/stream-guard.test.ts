import { describe, it, expect } from "vitest";
import { detectRepetition } from "../src/stream-guard.js";

describe("detectRepetition", () => {
  it("detects a short repeating pattern", () => {
    const buffer = "hello world. hello world. hello world. ";
    const result = detectRepetition(buffer, 5, 100, 3);
    expect(result).toBeTruthy();
  });

  it("detects single-word repetition", () => {
    const buffer = "the the the the the the the the the the ";
    const result = detectRepetition(buffer, 3, 100, 3);
    expect(result).toBe("the ");
  });

  it("returns null for non-repeating text", () => {
    const buffer = "The quick brown fox jumps over the lazy dog. A different sentence follows.";
    const result = detectRepetition(buffer, 5, 100, 3);
    expect(result).toBeNull();
  });

  it("respects minLen — ignores patterns shorter than threshold", () => {
    // "ab" repeats but minLen=5 should skip it
    const buffer = "abababababab";
    const result = detectRepetition(buffer, 5, 100, 3);
    expect(result).toBeNull();
  });

  it("respects minRepeats", () => {
    // Only 2 repeats, require 3
    const buffer = "some unique prefix. repeated chunk. repeated chunk. ";
    const result = detectRepetition(buffer, 5, 100, 3);
    expect(result).toBeNull();
  });

  it("detects longer degenerate patterns", () => {
    const phrase = "This is a degenerate output that keeps repeating itself. ";
    const buffer = phrase.repeat(5);
    const result = detectRepetition(buffer, 10, 200, 3);
    expect(result).toBe(phrase);
  });

  it("handles buffer shorter than minLen * minRepeats", () => {
    const buffer = "short";
    const result = detectRepetition(buffer, 10, 100, 3);
    expect(result).toBeNull();
  });
});
