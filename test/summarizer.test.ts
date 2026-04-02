import { describe, it, expect } from "vitest";
import { encode } from "gpt-tokenizer";
import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";
import { summarise } from "../src/summarizer.js";

const OUTPUT_DIR = path.join(import.meta.dirname, "output");
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const SHORT_TEXT = `
Artificial intelligence has transformed how businesses operate. Companies now use AI for
customer service, data analysis, and product recommendations. Machine learning models can
process vast amounts of data faster than humans, identifying patterns that would otherwise
go unnoticed. However, concerns about bias, privacy, and job displacement remain significant
challenges that the industry must address.
`;

const MEDIUM_TEXT = `
The history of space exploration spans decades of human achievement. In 1957, the
Soviet Union launched Sputnik, the first artificial satellite, marking the beginning of the
Space Age. This was followed by Yuri Gagarin's historic flight in 1961, making him the first
human in space. The United States responded with the Apollo program, culminating in Neil
Armstrong and Buzz Aldrin walking on the Moon in July 1969.

After the Moon landings, focus shifted to building space stations. Skylab, launched in 1973,
was America's first space station. The Soviet Union operated several Salyut stations before
launching Mir in 1986, which remained in orbit until 2001. The International Space Station,
a collaboration among multiple nations, has been continuously occupied since November 2000.

In recent years, private companies have revolutionised space travel. SpaceX, founded by Elon
Musk, developed reusable rockets, dramatically reducing launch costs. Blue Origin, Jeff Bezos's
company, focuses on suborbital tourism and heavy-lift vehicles. These companies, along with
others like Rocket Lab and Virgin Galactic, are making space more accessible than ever before.

Looking ahead, NASA's Artemis program aims to return humans to the Moon, while Mars remains
the ultimate goal for human exploration. China and India have also emerged as major space powers,
with successful lunar and Mars missions. Future plans include potential asteroid mining and the establishment of permanent lunar bases.
`;

const LONG_TEXT = `
Climate change represents one of the most pressing challenges facing humanity in the 21st century.
The Earth's average temperature has risen more than 1 degree Celsius above pre-industrial
levels, driven primarily by the burning of fossil fuels and deforestation. This warming has
triggered a cascade of environmental changes that affect every corner of the globe.

The Arctic is warming at nearly four times the global average rate. Sea ice extent has declined
dramatically, with summer ice coverage shrinking by approximately 13% per decade since satellite
observations began in 1979. This ice loss creates a feedback loop: as reflective ice melts, darker
ocean water absorbs more solar energy, accelerating further warming. Permafrost, ground that has
been frozen for thousands of years, is beginning to thaw, releasing stored methane and carbon
dioxide—potent greenhouse gases that could amplify warming beyond current projections.

Rising global temperatures are reshaping weather patterns worldwide. Extreme heat events that once
occurred perhaps once in 50 years are now happening roughly once per decade. Hurricanes and typhoons
are intensifying, carrying more moisture and producing heavier rainfall. Drought conditions are
expanding in subtropical regions, threatening water supplies and agricultural productivity. The
Mediterranean, parts of South America, and southern Africa face particularly severe drying trends.

Ocean systems are undergoing profound changes. Sea levels have risen about 20 centimetres since
1900, with the rate of rise accelerating. Thermal expansion of seawater and melting of land-based
ice sheets in Greenland and Antarctica are the primary drivers. Coastal communities worldwide face
increasing flood risks, and several Pacific island nations confront the possibility of becoming
uninhabitable within decades. Ocean acidification, caused by absorption of excess atmospheric CO2,
threatens coral reefs and shellfish populations, with cascading effects through marine food webs.

The impacts on biodiversity are severe and accelerating. Species are shifting their ranges poleward
and to higher elevations as temperatures rise. Coral reefs, which support roughly 25% of all marine
species, have experienced repeated mass bleaching events. The Great Barrier Reef alone has suffered
multiple mass bleaching events since 2016. On land, changing seasons disrupt the timing of ecological
events—flowers bloom before pollinators emerge, migratory birds arrive to find food sources depleted.

Agriculture faces both opportunities and threats from climate change. While some northern regions may
experience longer growing seasons, tropical and subtropical agricultural zones face declining yields
for staple crops including wheat, rice, and maize. Water scarcity compounds these challenges, as
glacial meltwater that feeds major river systems diminishes. The Himalayan glaciers, which provide
water for nearly two billion people across Asia, could lose a third of their ice by 2100 even under
optimistic emissions scenarios.

Addressing climate change requires action on multiple fronts. The Paris Agreement of 2015 established
a framework for limiting warming to well below 2°C, preferably 1.5°C, above pre-industrial levels.
Achieving these targets demands rapid decarbonisation of energy systems, with renewable sources like
solar and wind replacing fossil fuels. Transportation must shift toward electrification, buildings
need improved efficiency, and industrial processes require transformation. Carbon capture technologies,
while still developing, may play a role in removing accumulated atmospheric CO2.

Economic analysis increasingly shows that the costs of inaction far exceed the costs of mitigation.
The transition to clean energy creates new industries and employment opportunities, even as fossil
fuel sectors contract. Climate finance mechanisms aim to support developing nations in both reducing
emissions and adapting to unavoidable changes. Green bonds, carbon pricing, and climate-focused
investment funds are channelling growing sums toward sustainable development.

Individual actions, while important, cannot substitute for systemic change. Policy frameworks that
set emissions reduction targets, phase out fossil fuel subsidies, protect forests, and incentivise
clean technology adoption are essential. International cooperation remains crucial, as climate change
respects no borders. The decisions made in the coming years will largely determine whether humanity can
limit warming to manageable levels or face increasingly catastrophic consequences.
`;

function createClient(): OpenAI {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY not set — copy .env.example to .env");
  }
  return new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });
}

const cases = [
  { label: "short",        text: SHORT_TEXT,  min: 30,  max: 60  },
  { label: "short-expand", text: SHORT_TEXT,  min: 100, max: 150 },
  { label: "medium",       text: MEDIUM_TEXT, min: 80,  max: 120 },
  { label: "long",         text: LONG_TEXT,   min: 100, max: 200 },
  { label: "long-shrink",  text: LONG_TEXT,   min: 30,  max: 60  },
];

describe("summarise - integration", () => {
  const client = createClient();

  for (const { label, text, min, max } of cases) {
    it(`[${label}] summarises to ${min}–${max} tokens`, async () => {
      const inputTokens = encode(text).length;
      const result = await summarise(client, text, min, max, { verbose: true });
      console.log(`  [${label}] input: ${inputTokens} → target: ${min}–${max} → output: ${result.tokens}, ${result.attempts} attempts, ${result.usage.input} in + ${result.usage.output} out = ${result.usage.total} total, ${(result.durationMs / 1000).toFixed(1)}s, ${result.outputTokensPerSecond.toFixed(1)} tokens/s`);

      const log = [
        `Label: ${label}`,
        `Input tokens: ${inputTokens}`,
        `Target: ${min}–${max} tokens`,
        `Output tokens: ${result.tokens}`,
        `Within range: ${result.withinRange}`,
        `Attempts: ${result.attempts}`,
        `API usage: ${result.usage.input} input + ${result.usage.output} output = ${result.usage.total} total`,
        `Duration: ${(result.durationMs / 1000).toFixed(1)}s`,
        `Output rate: ${result.outputTokensPerSecond.toFixed(1)} tokens/s`,
        "",
        result.summary,
      ].join("\n");
      fs.writeFileSync(path.join(OUTPUT_DIR, `${label.replace(/→/g, "-")}.txt`), log);

      expect(result.summary.length).toBeGreaterThan(0);
      expect(result.withinRange).toBe(true);
      expect(result.tokens).toBeGreaterThanOrEqual(min);
      expect(result.tokens).toBeLessThanOrEqual(max);
    }, 120_000);
  }

  it("[long-shrink-budget] summarises with context budget and 10 attempts", async () => {
    const min = 10, max = 30;
    const maxFitAttempts = 10;
    const contextBudget = 1500;
    const inputTokens = encode(LONG_TEXT).length;
    const result = await summarise(client, LONG_TEXT, min, max, {
      verbose: true,
      maxFitAttempts,
      contextBudget,
    });
    console.log(`  [long-shrink-budget] input: ${inputTokens} → target: ${min}–${max} → output: ${result.tokens}, ${result.attempts} attempts, ${result.usage.input} in + ${result.usage.output} out = ${result.usage.total} total, ${(result.durationMs / 1000).toFixed(1)}s, ${result.outputTokensPerSecond.toFixed(1)} tokens/s`);

    const log = [
      `Label: long-shrink-budget`,
      `Input tokens: ${inputTokens}`,
      `Target: ${min}–${max} tokens`,
      `Max attempts: ${maxFitAttempts}`,
      `Context budget: ${contextBudget}`,
      `Output tokens: ${result.tokens}`,
      `Within range: ${result.withinRange}`,
      `Attempts: ${result.attempts}`,
      `API usage: ${result.usage.input} input + ${result.usage.output} output = ${result.usage.total} total`,
      `Duration: ${(result.durationMs / 1000).toFixed(1)}s`,
      `Output rate: ${result.outputTokensPerSecond.toFixed(1)} tokens/s`,
      "",
      result.summary,
    ].join("\n");
    fs.writeFileSync(path.join(OUTPUT_DIR, "long-short-budget.txt"), log);

    expect(result.summary.length).toBeGreaterThan(0);
    expect(result.withinRange).toBe(true);
    expect(result.tokens).toBeGreaterThanOrEqual(min);
    expect(result.tokens).toBeLessThanOrEqual(max);
  }, 300_000);
});

describe("token counting consistency", () => {
  it("encode produces consistent counts for same input", () => {
    const text = "The quick brown fox jumps over the lazy dog.";
    const count1 = encode(text).length;
    const count2 = encode(text).length;
    expect(count1).toBe(count2);
    expect(count1).toBeGreaterThan(0);
  });

  it("token count scales with text length", () => {
    const short = encode(SHORT_TEXT).length;
    const medium = encode(MEDIUM_TEXT).length;
    const long = encode(LONG_TEXT).length;
    expect(medium).toBeGreaterThan(short);
    expect(long).toBeGreaterThan(medium);
  });
});
