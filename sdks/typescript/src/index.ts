/**
 * Agentic Research Search Engine - TypeScript SDK
 *
 * Task 17.3: Generated TypeScript SDK from OpenAPI with typed methods and async iterators.
 * Task 17.4: SDK error mapping — non-2xx, timeout, connection failure → typed exception.
 * Task 17.5: SDK bearer header injection from configured key.
 *
 * Validates: Requirements R16.1, R16.2, R16.3, R16.5
 */

// ---------------------------------------------------------------------------
// Error types (Task 17.4)
// ---------------------------------------------------------------------------

export class SDKError extends Error {
  public readonly statusCode?: number;
  public readonly errorCode?: string;
  public readonly requestId?: string;

  constructor(
    message: string,
    options?: { statusCode?: number; errorCode?: string; requestId?: string }
  ) {
    super(message);
    this.name = "SDKError";
    this.statusCode = options?.statusCode;
    this.errorCode = options?.errorCode;
    this.requestId = options?.requestId;
  }
}

export class APIError extends SDKError {
  constructor(
    message: string,
    options?: { statusCode?: number; errorCode?: string; requestId?: string }
  ) {
    super(message, options);
    this.name = "APIError";
  }
}

export class TimeoutError extends SDKError {
  constructor(message: string, options?: { requestId?: string }) {
    super(message, { ...options, statusCode: undefined, errorCode: undefined });
    this.name = "TimeoutError";
  }
}

export class ConnectionError extends SDKError {
  constructor(message: string, options?: { requestId?: string }) {
    super(message, { ...options, statusCode: undefined, errorCode: undefined });
    this.name = "ConnectionError";
  }
}

export class ParseError extends SDKError {
  constructor(
    message: string,
    options?: { statusCode?: number; requestId?: string }
  ) {
    super(message, { ...options, errorCode: undefined });
    this.name = "ParseError";
  }
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

export type SearchMode = "neural" | "keyword" | "hybrid";

export interface ProvenanceInfo {
  credibility_score: number;
  ai_generated_likelihood: number;
  scored_at: string;
}

export interface SearchResult {
  document_id: string;
  url: string;
  title: string;
  score: number;
  published_at: string | null;
  provenance: ProvenanceInfo;
}

export interface SearchResponse {
  results: SearchResult[];
  warnings?: Array<{ code: string; step?: string }>;
  index_version?: number;
}

export interface Citation {
  document_id: string;
  version: number;
  answer_start: number;
  answer_end: number;
  source_start: number;
  source_end: number;
}

export interface ContentEntry {
  document_id: string;
  version?: number;
  cleaned_text?: string;
  highlights?: Array<{ start: number; end: number }>;
  summary?: string;
  error?: { code: string; message: string };
}

export interface ContentsResponse {
  results: ContentEntry[];
}

export interface Session {
  session_id: string;
  created_at: string;
  retention_days: number;
  expires_at?: string;
}

export interface PipelineStepDef {
  type: "filter" | "reranker" | "transform";
  registry_name: string;
  config?: Record<string, unknown>;
  timeout_ms?: number;
}

export interface Pipeline {
  pipeline_id: string;
  name: string;
  steps: PipelineStepDef[];
  created_at: string;
}

export interface ResearchJob {
  job_id: string;
  state: string;
  created_at: string;
  report?: Record<string, unknown>;
  citations?: Citation[];
}

export interface StreamEvent {
  event_type: string;
  data: Record<string, unknown>;
  event_id?: number;
}

// ---------------------------------------------------------------------------
// Request options
// ---------------------------------------------------------------------------

export interface SearchOptions {
  query: string;
  mode?: SearchMode;
  num_results?: number;
  filters?: string;
  pipeline_id?: string;
  min_credibility?: number;
  max_ai_generated_likelihood?: number;
}

export interface FindSimilarOptions {
  url: string;
  num_results?: number;
  filters?: string;
  min_credibility?: number;
  max_ai_generated_likelihood?: number;
}

export interface ContentsOptions {
  document_ids: string[];
  highlights?: boolean;
  query?: string;
  summary?: boolean;
}

export interface AnswerOptions {
  query: string;
  mode?: SearchMode;
  num_results?: number;
  stream?: boolean;
  session_id?: string;
}

export interface ResearchOptions {
  research_goal: string;
  output_schema?: Record<string, unknown>;
  session_id?: string;
  max_steps?: number;
  max_duration_ms?: number;
  max_tool_calls?: number;
}

// ---------------------------------------------------------------------------
// HTTP client interface (for dependency injection / testing)
// ---------------------------------------------------------------------------

export interface HttpResponse {
  status: number;
  headers: Record<string, string>;
  json(): Promise<unknown>;
  body?: ReadableStream<Uint8Array> | AsyncIterable<string>;
}

export interface HttpClient {
  request(
    method: string,
    url: string,
    options: {
      headers: Record<string, string>;
      body?: string;
      signal?: AbortSignal;
    }
  ): Promise<HttpResponse>;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export interface ClientConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
  httpClient?: HttpClient;
}

export class AgenticResearchClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeout: number;
  private readonly httpClient?: HttpClient;

  constructor(config: ClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30000;
    this.httpClient = config.httpClient;
  }

  /** Build headers with bearer token injection (R16.5). */
  private get headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    };
  }

  private buildUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private async request(
    method: string,
    path: string,
    options?: { body?: Record<string, unknown>; expectedStatus?: number }
  ): Promise<Record<string, unknown>> {
    const url = this.buildUrl(path);
    const expectedStatus = options?.expectedStatus ?? 200;

    let response: HttpResponse;

    if (this.httpClient) {
      response = await this.httpClient.request(method, url, {
        headers: this.headers,
        body: options?.body ? JSON.stringify(options.body) : undefined,
      });
    } else {
      // Use native fetch
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        const fetchResponse = await fetch(url, {
          method,
          headers: this.headers,
          body: options?.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        response = {
          status: fetchResponse.status,
          headers: Object.fromEntries(fetchResponse.headers.entries()),
          json: () => fetchResponse.json(),
          body: fetchResponse.body ?? undefined,
        };
      } catch (err: unknown) {
        clearTimeout(timeoutId);
        if (err instanceof Error && err.name === "AbortError") {
          throw new TimeoutError(`Request timed out after ${this.timeout}ms`);
        }
        throw new ConnectionError(
          `Connection failed: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    }

    const requestId = response.headers["x-request-id"] ?? undefined;

    if (
      response.status !== expectedStatus &&
      (response.status < 200 || response.status >= 300)
    ) {
      let errorCode: string | undefined;
      let message = `HTTP ${response.status}`;
      try {
        const body = (await response.json()) as Record<string, unknown>;
        const error = body.error as Record<string, unknown> | undefined;
        if (error) {
          errorCode = error.code as string | undefined;
          message = (error.message as string) ?? message;
        }
      } catch {
        // ignore parse errors on error responses
      }

      throw new APIError(message, {
        statusCode: response.status,
        errorCode,
        requestId,
      });
    }

    try {
      return (await response.json()) as Record<string, unknown>;
    } catch (err) {
      throw new ParseError(
        `Failed to parse response: ${err instanceof Error ? err.message : String(err)}`,
        { statusCode: response.status, requestId }
      );
    }
  }

  // -------------------------------------------------------------------------
  // Search endpoints
  // -------------------------------------------------------------------------

  async search(options: SearchOptions): Promise<SearchResponse> {
    const body: Record<string, unknown> = {
      query: options.query,
      mode: options.mode ?? "hybrid",
      num_results: options.num_results ?? 10,
    };
    if (options.filters !== undefined) body.filters = options.filters;
    if (options.pipeline_id !== undefined) body.pipeline_id = options.pipeline_id;
    if (options.min_credibility !== undefined)
      body.min_credibility = options.min_credibility;
    if (options.max_ai_generated_likelihood !== undefined)
      body.max_ai_generated_likelihood = options.max_ai_generated_likelihood;

    const data = await this.request("POST", "/search", { body });
    return data as unknown as SearchResponse;
  }

  async findSimilar(options: FindSimilarOptions): Promise<SearchResponse> {
    const body: Record<string, unknown> = {
      url: options.url,
      num_results: options.num_results ?? 10,
    };
    if (options.filters !== undefined) body.filters = options.filters;
    if (options.min_credibility !== undefined)
      body.min_credibility = options.min_credibility;
    if (options.max_ai_generated_likelihood !== undefined)
      body.max_ai_generated_likelihood = options.max_ai_generated_likelihood;

    const data = await this.request("POST", "/find_similar", { body });
    return data as unknown as SearchResponse;
  }

  async contents(options: ContentsOptions): Promise<ContentsResponse> {
    const body: Record<string, unknown> = {
      document_ids: options.document_ids,
      highlights: options.highlights ?? false,
      summary: options.summary ?? false,
    };
    if (options.query !== undefined) body.query = options.query;

    const data = await this.request("POST", "/contents", { body });
    return data as unknown as ContentsResponse;
  }

  // -------------------------------------------------------------------------
  // Answer endpoint with streaming (R16.2)
  // -------------------------------------------------------------------------

  async *answer(options: AnswerOptions): AsyncGenerator<StreamEvent> {
    const body: Record<string, unknown> = {
      query: options.query,
      mode: options.mode ?? "hybrid",
      num_results: options.num_results ?? 10,
      stream: options.stream ?? true,
    };
    if (options.session_id !== undefined) body.session_id = options.session_id;

    for await (const event of this.streamSSE("/answer", body)) {
      yield event;
      if (event.event_type === "done" || event.event_type === "error") {
        return;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Research endpoints
  // -------------------------------------------------------------------------

  async createResearch(options: ResearchOptions): Promise<string> {
    const body: Record<string, unknown> = {
      research_goal: options.research_goal,
    };
    if (options.output_schema !== undefined)
      body.output_schema = options.output_schema;
    if (options.session_id !== undefined) body.session_id = options.session_id;
    if (options.max_steps !== undefined) body.max_steps = options.max_steps;
    if (options.max_duration_ms !== undefined)
      body.max_duration_ms = options.max_duration_ms;
    if (options.max_tool_calls !== undefined)
      body.max_tool_calls = options.max_tool_calls;

    const data = await this.request("POST", "/research", {
      body,
      expectedStatus: 201,
    });
    return data.job_id as string;
  }

  async getResearchJob(jobId: string): Promise<ResearchJob> {
    const data = await this.request("GET", `/research/${jobId}`);
    return data as unknown as ResearchJob;
  }

  async *researchEvents(
    jobId: string,
    options?: { lastEventId?: number }
  ): AsyncGenerator<StreamEvent> {
    for await (const event of this.streamSSEGet(
      `/research/${jobId}/events`,
      options?.lastEventId
    )) {
      yield event;
      if (event.event_type === "done" || event.event_type === "error") {
        return;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Session endpoints
  // -------------------------------------------------------------------------

  async createSession(retentionDays: number = 14): Promise<Session> {
    const data = await this.request("POST", "/sessions", {
      body: { retention_days: retentionDays },
      expectedStatus: 201,
    });
    return data as unknown as Session;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.request("DELETE", `/sessions/${sessionId}`, {
      expectedStatus: 204,
    });
  }

  // -------------------------------------------------------------------------
  // Pipeline endpoints
  // -------------------------------------------------------------------------

  async createPipeline(
    name: string,
    steps: PipelineStepDef[]
  ): Promise<Pipeline> {
    const data = await this.request("POST", "/pipelines", {
      body: { name, steps },
      expectedStatus: 201,
    });
    return data as unknown as Pipeline;
  }

  async getPipeline(pipelineId: string): Promise<Pipeline> {
    const data = await this.request("GET", `/pipelines/${pipelineId}`);
    return data as unknown as Pipeline;
  }

  async deletePipeline(pipelineId: string): Promise<void> {
    await this.request("DELETE", `/pipelines/${pipelineId}`, {
      expectedStatus: 204,
    });
  }

  // -------------------------------------------------------------------------
  // SSE streaming helpers
  // -------------------------------------------------------------------------

  private async *streamSSE(
    path: string,
    body: Record<string, unknown>
  ): AsyncGenerator<StreamEvent> {
    const url = this.buildUrl(path);
    const headers = { ...this.headers, Accept: "text/event-stream" };

    if (this.httpClient) {
      const response = await this.httpClient.request("POST", url, {
        headers,
        body: JSON.stringify(body),
      });
      if (response.body) {
        for await (const event of this.parseSSEStream(
          response.body as AsyncIterable<string>
        )) {
          yield event;
        }
      }
    } else {
      const fetchResponse = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (fetchResponse.body) {
        const reader = fetchResponse.body.getReader();
        const decoder = new TextDecoder();
        const lines = this.readLines(reader, decoder);
        for await (const event of this.parseSSELines(lines)) {
          yield event;
        }
      }
    }
  }

  private async *streamSSEGet(
    path: string,
    lastEventId?: number
  ): AsyncGenerator<StreamEvent> {
    const url = this.buildUrl(path);
    const headers: Record<string, string> = {
      ...this.headers,
      Accept: "text/event-stream",
    };
    if (lastEventId !== undefined) {
      headers["Last-Event-ID"] = String(lastEventId);
    }

    if (this.httpClient) {
      const response = await this.httpClient.request("GET", url, { headers });
      if (response.body) {
        for await (const event of this.parseSSEStream(
          response.body as AsyncIterable<string>
        )) {
          yield event;
        }
      }
    } else {
      const fetchResponse = await fetch(url, { method: "GET", headers });
      if (fetchResponse.body) {
        const reader = fetchResponse.body.getReader();
        const decoder = new TextDecoder();
        const lines = this.readLines(reader, decoder);
        for await (const event of this.parseSSELines(lines)) {
          yield event;
        }
      }
    }
  }

  private async *readLines(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    decoder: { decode(input?: Uint8Array, options?: { stream?: boolean }): string }
  ): AsyncGenerator<string> {
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        yield line;
      }
    }
    if (buffer) yield buffer;
  }

  private async *parseSSELines(
    lines: AsyncIterable<string>
  ): AsyncGenerator<StreamEvent> {
    let eventType = "";
    let eventId: number | undefined;
    let dataLines: string[] = [];

    for await (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("id:")) {
        const id = parseInt(line.slice(3).trim(), 10);
        eventId = isNaN(id) ? undefined : id;
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      } else if (line.trim() === "" && (eventType || dataLines.length > 0)) {
        const dataStr = dataLines.join("\n");
        let data: Record<string, unknown>;
        try {
          data = dataStr ? JSON.parse(dataStr) : {};
        } catch {
          data = { raw: dataStr };
        }
        yield { event_type: eventType, data, event_id: eventId };
        eventType = "";
        eventId = undefined;
        dataLines = [];
      }
    }
  }

  private async *parseSSEStream(
    stream: AsyncIterable<string>
  ): AsyncGenerator<StreamEvent> {
    for await (const event of this.parseSSELines(stream)) {
      yield event;
    }
  }
}
