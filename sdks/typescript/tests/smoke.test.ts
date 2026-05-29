import { describe, it, expect } from "vitest";

describe("SDK smoke test", () => {
  it("should import the SDK entry point", async () => {
    const sdk = await import("../src/index");
    expect(sdk).toBeDefined();
  });

  it("should have vitest configured correctly", () => {
    expect(true).toBe(true);
  });

  it("should support async tests", async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });
});
