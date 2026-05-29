import { describe, it, expect } from "vitest";
import * as fc from "fast-check";

describe("Property-based testing with fast-check", () => {
  it("property: normalized scores are always in [0.0, 1.0]", () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 100 }), (scorePct) => {
        const normalized = scorePct / 100.0;
        expect(normalized).toBeGreaterThanOrEqual(0.0);
        expect(normalized).toBeLessThanOrEqual(1.0);
      }),
    );
  });

  it("property: string encode/decode is a round-trip", () => {
    fc.assert(
      fc.property(fc.string({ minLength: 1, maxLength: 100 }), (s) => {
        const encoded = new TextEncoder().encode(s);
        const decoded = new TextDecoder().decode(encoded);
        expect(decoded).toBe(s);
      }),
    );
  });

  it("property: sorting scores maintains non-increasing order", () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0.0, max: 1.0, noNaN: true }), {
          minLength: 1,
          maxLength: 50,
        }),
        (scores) => {
          const sorted = [...scores].sort((a, b) => b - a);
          for (let i = 0; i < sorted.length - 1; i++) {
            expect(sorted[i]).toBeGreaterThanOrEqual(sorted[i + 1]);
          }
        },
      ),
    );
  });

  it("property: UUID format is valid RFC 4122", () => {
    const uuidRegex =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

    fc.assert(
      fc.property(fc.uuid(), (id) => {
        // fast-check generates valid RFC 4122 UUIDs (any version)
        expect(id).toMatch(uuidRegex);
      }),
    );
  });
});
