# Requirements Document

## Introduction

The Agentic Research Search Engine is an API-first, multi-tenant SaaS that goes beyond traditional neural search products (e.g., Exa) by treating *agentic deep research* as a first-class primitive. Developers building AI agents call a small set of REST and streaming endpoints (and an MCP tool-calling surface) to: (1) search a continuously crawled web index using neural, keyword, and hybrid retrieval, (2) retrieve cleaned page contents with highlights and summaries, (3) generate streaming answers with verifiable citations, and (4) launch multi-hop research jobs that plan, retrieve, reason, and produce structured reports with full provenance.

The MVP is anchored on five differentiators relative to Exa:

1. **Agentic deep research** with explicit, inspectable plans, multi-hop reasoning, and tool-use loops.
2. **Streaming answers with live citations** rather than batch-only responses.
3. **Programmable retrieval pipelines** (user-defined filters, rerankers, transforms) composed via a typed DSL.
4. **Provenance and credibility scoring**, including AI-generated content detection and anti-SEO spam signals.
5. **Persistent research sessions** with memory that carry context across queries within a tenant.

The system is delivered as a managed multi-tenant SaaS with REST, SSE, WebSocket, and MCP interfaces, plus first-class Python and TypeScript SDKs. Crawling respects `robots.txt`, content licensing, and site-owner opt-outs. The platform enforces tenant isolation, audit logging, rate limiting, PII redaction, and configurable data retention from day one.

## Glossary

- **Search_Engine**: The top-level system; the union of all subsystems described below.
- **Crawler**: The subsystem that fetches public web pages, respects `robots.txt`, and produces raw documents.
- **Indexer**: The subsystem that converts raw documents into searchable index entries (lexical, vector, and metadata).
- **Retriever**: The query-time subsystem that returns a ranked set of documents for a query using neural, keyword, or hybrid retrieval.
- **Pipeline_Engine**: The subsystem that executes a tenant-defined retrieval pipeline composed of filters, rerankers, and transforms over Retriever output.
- **Answer_Engine**: The subsystem that synthesizes a natural-language answer with inline citations from a set of retrieved documents.
- **Research_Agent**: The subsystem that plans and executes multi-hop research jobs, calling Retriever, Pipeline_Engine, and Answer_Engine in a tool-use loop.
- **Session_Store**: The subsystem that persists Research_Agent state, memory, and intermediate artifacts across requests within a single tenant.
- **Provenance_Scorer**: The subsystem that assigns each indexed document a credibility score and an AI-generated-content likelihood score.
- **Query_Filter_DSL**: The textual filter language used to constrain searches (domains, dates, categories, custom metadata).
- **Query_Filter_Parser**: The component that parses Query_Filter_DSL strings into Filter_AST values.
- **Query_Filter_Printer**: The component that serializes Filter_AST values back into Query_Filter_DSL strings.
- **Filter_AST**: The in-memory abstract syntax tree representation of a parsed filter expression.
- **API_Gateway**: The subsystem exposing REST, Server-Sent Events (SSE), and WebSocket endpoints to tenants.
- **MCP_Server**: The Model Context Protocol surface that exposes Search_Engine capabilities as LLM tool calls.
- **Auth_Service**: The subsystem that authenticates API keys, enforces tenant isolation, and applies rate limits.
- **Audit_Log**: The append-only log capturing every privileged action (data access, configuration change, deletion).
- **PII_Redactor**: The component that detects and redacts personally identifiable information from queries, stored content, and logs.
- **Tenant**: An isolated customer account; all resources (keys, sessions, pipelines, audit entries) are scoped to a tenant.
- **Citation**: A structured reference linking a span of generated text to a specific URL, document version, and character offset range.
- **Research_Plan**: A structured, serializable description of the steps a Research_Agent will execute for a research job.

## Requirements

### Requirement 1: Ethical Web Crawling

**User Story:** As a platform operator, I want the Crawler to fetch public web content while respecting site-owner directives, so that the system stays compliant with robots policies and content licensing expectations.

#### Acceptance Criteria

1. WHEN the Crawler intends to fetch a URL on a host and no fresh `robots.txt` for that host is cached, THE Crawler SHALL fetch the host's `robots.txt` over a request bounded by a 10-second timeout and SHALL cache the result for at most 24 hours before re-fetching.
2. IF the cached or freshly fetched `robots.txt` for a host disallows a URL for the configured user agent, THEN THE Crawler SHALL skip the URL and SHALL record a `disallowed_by_robots` reason in the Audit_Log capturing the URL and the matched directive.
3. IF fetching the host's `robots.txt` fails with a network error, a timeout, or an HTTP 5xx status, THEN THE Crawler SHALL skip the URL, SHALL NOT issue the content fetch, and SHALL record a `robots_unavailable` reason in the Audit_Log.
4. WHILE the Crawler is fetching from a single host, THE Crawler SHALL cap concurrent in-flight requests per host at a configured maximum in the integer range [1, 8] with a default of 2.
5. WHEN the Crawler issues sequential requests to a single host, THE Crawler SHALL space consecutive requests by at least the host's `Crawl-Delay` directive when present and by at least 1 second otherwise.
6. WHERE a site owner has submitted an opt-out request through the published opt-out endpoint, THE Crawler SHALL exclude all URLs on that site's registrable domain from any crawl initiated more than 24 hours after the opt-out's recorded acceptance timestamp and SHALL record a `domain_opted_out` reason in the Audit_Log for each subsequent skip.
7. WHEN the Crawler stores a fetched document, THE Crawler SHALL record alongside the content the fetch timestamp in UTC, the HTTP status code, the response `Content-Type`, and the canonical source URL.

### Requirement 2: Indexing and Freshness

**User Story:** As an API consumer, I want indexed content to reflect recent web changes, so that my agents can answer questions about recently published material.

#### Acceptance Criteria

1. WHEN the Crawler emits a fetched document with HTTP status in the 2xx range, THE Indexer SHALL produce a searchable index entry containing lexical, vector, and metadata fields available to Retriever within 60 minutes of fetch completion at the 95th percentile and within 4 hours at the 99th percentile.
2. THE Indexer SHALL re-crawl every URL designated as a "priority source" through the platform's source configuration at least once per rolling 24-hour window measured in UTC.
3. WHEN a document is re-indexed and its content hash, computed over the document's cleaned text content, differs from the previously indexed version's hash, THE Indexer SHALL preserve the `document_id` assigned at first ingest unchanged and SHALL increment the `version` field by exactly 1.
4. WHEN the Indexer receives a document whose content hash, computed over the document's cleaned text content, matches the most recently indexed version's hash, THE Indexer SHALL update only the `last_seen_at` timestamp in UTC and SHALL NOT modify `version`, `document_id`, or stored content.
5. IF the Indexer fails to index a document after 3 retry attempts spaced by at least 60 seconds between consecutive attempts, THEN THE Indexer SHALL move the document to a dead-letter queue, SHALL emit an `index_failure` event in the Audit_Log capturing the `document_id` (or source URL when no `document_id` exists), the failure reason, and a UTC timestamp, and SHALL NOT automatically retry the document until manual reprocessing is requested.

### Requirement 3: Neural, Keyword, and Hybrid Search API

**User Story:** As a developer building an AI agent, I want a single search endpoint that supports neural, keyword, and hybrid retrieval, so that I can choose the best strategy per query.

#### Acceptance Criteria

1. WHEN the API_Gateway receives a `POST /v1/search` request whose `query` field, after trimming surrounding whitespace, contains between 1 and 2,048 Unicode code points and whose `mode` is exactly one of `neural`, `keyword`, or `hybrid`, THE Retriever SHALL return between 0 and `num_results` ranked documents, where `num_results` defaults to 10 when omitted and is capped at 100.
2. WHEN the API_Gateway receives a `POST /v1/search` request whose `(tenant_id, query, mode, filters, pipeline_id, num_results)` tuple matches an identical tuple served within the prior 5 minutes (a "warm cache hit"), THE API_Gateway SHALL respond within 800 milliseconds at the 95th percentile measured over a rolling 1-hour window with at least 100 samples.
3. WHEN the Retriever emits a search result, THE API_Gateway SHALL include for that result a `document_id`, `url`, `title`, `score` in the closed interval [0.0, 1.0] with results ordered by non-increasing `score`, `published_at` (an ISO 8601 UTC timestamp when known and `null` otherwise), and a `provenance` object containing `credibility_score` and `ai_generated_likelihood` each in the closed interval [0.0, 1.0] from Provenance_Scorer.
4. WHEN the same `query`, `mode`, `filters`, and `pipeline_id` are submitted twice against an unchanged index version, THE Retriever SHALL return results in identical order (deterministic ranking).
5. IF a `POST /v1/search` request specifies `num_results` greater than 100 or less than 0, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_num_results` error code and SHALL NOT invoke the Retriever.
6. IF a `POST /v1/search` request specifies a `mode` value other than `neural`, `keyword`, or `hybrid`, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_mode` error code and SHALL NOT invoke the Retriever.
7. IF a `POST /v1/search` request omits `query`, supplies a `query` whose trimmed length is 0 code points, or supplies a `query` whose length exceeds 2,048 code points, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_query` error code and SHALL NOT invoke the Retriever.

### Requirement 4: Find-Similar API

**User Story:** As a developer, I want to find pages semantically similar to a given URL, so that my agent can expand its evidence set from a known good source.

#### Acceptance Criteria

1. WHEN the API_Gateway receives a `POST /v1/find_similar` request with a URL that resolves to an indexed `document_id`, THE Retriever SHALL return between 0 and `num_results` documents ordered by strictly non-increasing semantic similarity to that document, where `num_results` defaults to 10 when omitted and is capped at 100, and each result contains `document_id`, `url`, `title`, `score`, `published_at`, and `provenance`.
2. THE Retriever SHALL exclude every version of the input document_id from `find_similar` results.
3. IF the canonicalized input URL (lowercased scheme and host, default port removed, fragment stripped, trailing slash normalized) is not present in the index, THEN THE API_Gateway SHALL return HTTP 404 with an `unknown_url` error code and SHALL NOT trigger an on-demand crawl.
4. WHEN `find_similar` is invoked twice with the same canonicalized URL, `num_results`, and filters against an unchanged index version, THE Retriever SHALL return results in identical order.
5. IF a `POST /v1/find_similar` request omits the URL, supplies a URL longer than 2,048 code points, or supplies a syntactically invalid URL, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_url` error code and SHALL NOT invoke the Retriever.
6. IF a `POST /v1/find_similar` request specifies `num_results` greater than 100 or less than 0, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_num_results` error code and SHALL NOT invoke the Retriever.

### Requirement 5: Content Retrieval with Highlights and Summaries

**User Story:** As a developer, I want to retrieve cleaned page text, highlights, and summaries, so that my agent has high-signal context to reason over.

#### Acceptance Criteria

1. WHEN the API_Gateway receives a `POST /v1/contents` request listing between 1 and 100 `document_id` values, THE Search_Engine SHALL return one result entry per requested `document_id` preserving request order, where each entry contains either cleaned text content or an `error` object with a stable string `code` and a human-readable `message` field.
2. WHERE the request includes `highlights: true` and a non-empty `query` field, THE Search_Engine SHALL return between 0 and 5 highlight spans per document, each expressed as a half-open `[start, end)` range of Unicode code-point offsets into the cleaned text satisfying `0 <= start < end <= length(cleaned_text)`.
3. WHERE the request includes `summary: true`, THE Answer_Engine SHALL return a summary per document containing between 1 and 512 model tokens.
4. THE Search_Engine SHALL include a `version` field on each returned document matching the indexed version used to produce the cleaned text, highlights, and summary.
5. IF a `POST /v1/contents` request lists fewer than 1 or more than 100 `document_id` values, THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_document_id_count` error code.
6. IF a `POST /v1/contents` request sets `highlights: true` without a non-empty `query` field, THEN THE API_Gateway SHALL reject the request with HTTP 400 and a `missing_highlight_query` error code.
7. IF a requested `document_id` is not present in the index or has been deleted from the index, THEN THE Search_Engine SHALL return an `error` object for that entry with code `document_not_found` and SHALL still return successful entries for the remaining requested `document_id` values.

### Requirement 6: Streaming Answer Generation with Citations

**User Story:** As a developer, I want answers streamed token-by-token with citations attached as they are produced, so that my agent can render live UI and verify sources incrementally.

#### Acceptance Criteria

1. WHEN the API_Gateway receives a `POST /v1/answer` request with `stream: true` over SSE or WebSocket, THE Answer_Engine SHALL emit the first `token` event within 3 seconds of request acceptance at the 95th percentile and SHALL continue emitting `token` events incrementally until generation completes.
2. WHEN the Answer_Engine emits a span of generated text supported by a retrieved document, THE Answer_Engine SHALL emit a `citation` event within 500 milliseconds of the supported span containing the supporting `document_id`, `version`, an answer-text offset range as a half-open `[answer_start, answer_end)` pair of Unicode code-point offsets into the answer text emitted so far, and a source offset range as a half-open `[source_start, source_end)` pair of Unicode code-point offsets into that document's cleaned text.
3. WHEN the Answer_Engine completes answer generation without error, THE Answer_Engine SHALL emit a final `done` event whose payload contains the full answer text and the complete set of Citation objects emitted during the stream.
4. WHEN the Answer_Engine emits a `citation` event, THE Answer_Engine SHALL ensure the cited `(document_id, version)` pair appears in the request's retrieval result set used for that answer.
5. IF the upstream model fails mid-stream or the Answer_Engine emits no `token` event for 30 consecutive seconds, THEN THE Answer_Engine SHALL emit exactly one `error` event with a stable error code drawn from a documented enumeration, SHALL emit no further `token` or `citation` events, and SHALL close the stream within 2 seconds of the `error` event.
6. IF the request's retrieval result set is empty, THEN THE Answer_Engine SHALL emit exactly one `error` event with the stable error code `no_sources_available`, SHALL emit no `token` or `citation` events, and SHALL close the stream within 2 seconds of the `error` event.

### Requirement 7: Agentic Deep Research

**User Story:** As a developer, I want to launch multi-hop research jobs that plan, search, read, and synthesize, so that my agent can answer questions that require chaining many queries.

#### Acceptance Criteria

1. WHEN the API_Gateway receives a `POST /v1/research` request whose `research_goal` field contains between 1 and 4,096 Unicode code points and whose optional `output_schema` is a syntactically valid JSON Schema document, THE Research_Agent SHALL return a `job_id` within 1 second at the 95th percentile and SHALL execute the job asynchronously.
2. WHEN a research job starts, THE Research_Agent SHALL emit a `plan_updated` event as the first event on the job's event stream containing a Research_Plan with between 1 and 32 steps, each labeled with a step type from the enumeration (`sub_query`, `retrieval`, `read`, `synthesis`), before any retrieval is performed.
3. WHEN a client requests `GET /v1/research/{job_id}/events` over SSE for a job belonging to the calling Tenant, THE Research_Agent SHALL stream `plan_updated`, `step_started`, `step_completed`, `citation`, and `report_chunk` events with strictly monotonically increasing `event_id` values, SHALL emit a terminal `done` or `error` event when the job ends, and SHALL replay events with `event_id` greater than the value supplied in `Last-Event-ID` on reconnection.
4. WHEN a research job completes successfully, THE Research_Agent SHALL produce a final report retrievable via `GET /v1/research/{job_id}` whose every factual claim is annotated with at least one Citation referencing an indexed document.
5. WHERE the request includes an `output_schema`, THE Research_Agent SHALL return a final report whose structured payload validates against that schema.
6. IF a research job exceeds its `max_steps`, `max_duration_ms`, or `max_tool_calls` budget (each defaulting to a tenant-level configured value when omitted from the request), THEN THE Research_Agent SHALL terminate the job with a terminal `error` event using stable code `budget_exceeded` and SHALL make available via `GET /v1/research/{job_id}` the partial report and citations gathered so far.
7. IF a `GET /v1/research/{job_id}` or `GET /v1/research/{job_id}/events` request supplies a `job_id` that does not belong to the calling Tenant, THEN THE API_Gateway SHALL respond with HTTP 404 and a `job_not_found` error code without disclosing whether the job exists in another Tenant.
8. IF a `POST /v1/research` request supplies a `research_goal` outside the 1–4,096 character bound or supplies an `output_schema` that is not a syntactically valid JSON Schema document, THEN THE API_Gateway SHALL reject the request with HTTP 400, an `invalid_research_request` error code, and SHALL NOT assign a `job_id` or start a job.

### Requirement 8: Persistent Research Sessions

**User Story:** As a developer, I want my agent's research context to persist across calls within a session, so that follow-up questions reuse prior findings without re-crawling.

#### Acceptance Criteria

1. WHEN a client creates a session via `POST /v1/sessions` with a `retention_days` value in the integer range [1, 90] (defaulting to 14 when omitted), THE Session_Store SHALL return a `session_id` unique across the calling Tenant and scoped to that Tenant.
2. WHEN a `POST /v1/research` or `POST /v1/answer` request includes a `session_id` that resolves to a non-expired session of the calling Tenant, THE Research_Agent SHALL incorporate up to the 50 most recent prior Citations and up to the 20 most recently retrieved unique `document_id` values from that session as additional retrieval context.
3. THE Session_Store SHALL only return session memory to requests whose authenticated Tenant matches the session's owning Tenant.
4. WHEN a session reaches its configured `retention_days` past creation, THE Session_Store SHALL delete all memory and artifacts for that session within 24 hours of expiry, SHALL stop incorporating that session's memory in any subsequent request, and SHALL emit a `session_expired` event to the Audit_Log capturing `session_id`, `tenant_id`, and the deletion timestamp in UTC.
5. IF a request supplies a `session_id` that does not exist, has expired and been deleted, or belongs to a different Tenant, THEN THE API_Gateway SHALL respond with HTTP 404 and a `session_not_found` error code with an identical response shape across all three cases without disclosing whether the session exists in another Tenant.

### Requirement 9: Programmable Retrieval Pipelines

**User Story:** As a developer, I want to declare a retrieval pipeline of filters, rerankers, and transforms, so that I can tune ranking for my domain without forking the engine.

#### Acceptance Criteria

1. WHEN a client submits `POST /v1/pipelines` with a pipeline definition containing between 1 and 20 steps in which every referenced filter, reranker, and transform name exists in the registry, THE Pipeline_Engine SHALL persist the pipeline scoped to the calling Tenant and SHALL return HTTP 201 with a generated `pipeline_id`.
2. IF a `POST /v1/pipelines` request references one or more filter, reranker, or transform names that do not exist in the registry, THEN THE Pipeline_Engine SHALL reject the request with HTTP 400 and an `unknown_pipeline_step` error code listing every offending step name and SHALL NOT persist the pipeline.
3. WHEN a `POST /v1/search` request includes a `pipeline_id` belonging to the calling Tenant, THE Pipeline_Engine SHALL execute the pipeline's steps in declared order and SHALL pass the output of each step as input to the next.
4. WHERE a pipeline definition lacks an explicit cross-type ordering, THE Pipeline_Engine SHALL apply filter steps before reranker steps and reranker steps before transform steps.
5. WHEN the same pipeline is executed twice against the same query, filters, and unchanged index version, THE Pipeline_Engine SHALL produce results in identical order.
6. IF a pipeline step exceeds its configured timeout (in the integer range [100, 30000] milliseconds with a default of 2,000), THEN THE Pipeline_Engine SHALL skip the timed-out step, SHALL append a `step_timeout` warning to the search response's `warnings` array identifying the step name, and SHALL continue using the input that was supplied to the timed-out step as the output of that step.
7. IF a `POST /v1/search` request supplies a `pipeline_id` that does not belong to the calling Tenant or does not exist, THEN THE API_Gateway SHALL respond with HTTP 404 and a `pipeline_not_found` error code without disclosing whether the pipeline exists in another Tenant.

### Requirement 10: Provenance and Credibility Scoring

**User Story:** As a developer, I want every result to carry credibility and AI-generation signals, so that my agent can prefer trustworthy sources and warn users about likely AI slop.

#### Acceptance Criteria

1. WHEN the Indexer indexes a document, THE Provenance_Scorer SHALL assign that document a `credibility_score` in the closed interval [0.0, 1.0], an `ai_generated_likelihood` in the closed interval [0.0, 1.0], and a `scored_at` timestamp in ISO 8601 UTC before the document becomes available to the Retriever.
2. THE Search_Engine SHALL include `credibility_score`, `ai_generated_likelihood`, and `scored_at` on every result returned by `/v1/search`, `/v1/find_similar`, and `/v1/contents`.
3. WHERE a search request specifies `min_credibility` as a number in the closed interval [0.0, 1.0], THE Retriever SHALL exclude documents whose `credibility_score` is strictly less than the specified threshold and SHALL include documents whose `credibility_score` equals the threshold.
4. WHERE a search request specifies `max_ai_generated_likelihood` as a number in the closed interval [0.0, 1.0], THE Retriever SHALL exclude documents whose `ai_generated_likelihood` is strictly greater than the specified threshold and SHALL include documents whose `ai_generated_likelihood` equals the threshold.
5. IF a search request specifies `min_credibility` or `max_ai_generated_likelihood` as a non-numeric value or a number outside the closed interval [0.0, 1.0], THEN THE API_Gateway SHALL reject the request with HTTP 400 and an `invalid_threshold` error code and SHALL NOT invoke the Retriever.
6. WHEN the Provenance_Scorer recomputes scores for an existing document, THE Provenance_Scorer SHALL preserve the document's `document_id` and `version` unchanged, SHALL update only the `credibility_score`, `ai_generated_likelihood`, and `scored_at` fields, and SHALL leave all other stored fields unchanged.

### Requirement 11: Query Filter DSL with Round-Trip Parsing

**User Story:** As a developer, I want a textual filter language that can be parsed, inspected, and re-printed losslessly, so that my tooling can build, store, and edit filters reliably.

#### Acceptance Criteria

1. WHEN the Query_Filter_Parser receives an input string of length between 1 and 16,384 Unicode code points that conforms to the Query_Filter_DSL grammar, THE Query_Filter_Parser SHALL produce a Filter_AST value representing that filter within 100 milliseconds on a single CPU core.
2. IF the Query_Filter_Parser receives an input string that is empty or contains only whitespace, THEN THE Query_Filter_Parser SHALL return an error with code `empty_input` and SHALL NOT produce a Filter_AST.
3. IF the Query_Filter_Parser receives an input string that is not syntactically valid Query_Filter_DSL, THEN THE Query_Filter_Parser SHALL return an error whose payload includes a 1-indexed line number, a 1-indexed column number pointing to the first offending character, and a human-readable description between 1 and 256 code points long, and SHALL NOT produce a partial Filter_AST.
4. IF the Query_Filter_Parser receives an input string longer than 16,384 code points or produces during parsing a Filter_AST that would exceed 32 levels of nesting or 1,024 leaf comparisons, THEN THE Query_Filter_Parser SHALL return an error with code `filter_too_large` and SHALL NOT produce a Filter_AST.
5. THE Query_Filter_Printer SHALL produce a Query_Filter_DSL string for every Filter_AST value such that parsing the printed string yields a Filter_AST equivalent to the input under the structural-equivalence rules in criterion 7 (round-trip property: `parse(print(ast)) ≡ ast`).
6. THE Query_Filter_Parser SHALL produce a Filter_AST such that printing and re-parsing yields a Filter_AST equivalent under the structural-equivalence rules in criterion 7 for every parseable input string (round-trip property: `parse(print(parse(s))) ≡ parse(s)` whenever `parse(s)` succeeds).
7. THE Filter_AST structural-equivalence relation SHALL hold when two ASTs share the same operator type, identical (case-sensitive) field references, identical literal values after numeric and ISO 8601 timestamp normalization, identical (order-insensitive) child-set membership for commutative operators (`and`, `or`, set membership), and identical child order for non-commutative operators (range and negation).
8. THE Query_Filter_DSL SHALL support, at minimum, equality, set membership (with set cardinality between 1 and 256), range with operators `lt`, `le`, `gt`, `ge`, conjunction (`and`), disjunction (`or`), and negation (`not`) over the fields `domain`, `url`, `published_at` (ISO 8601 timestamp literals), `language`, `category`, and `metadata.*` keys, where each metadata key segment is between 1 and 128 code points and each string literal is between 0 and 1,024 code points.

### Requirement 12: MCP Tool-Calling Interface

**User Story:** As an LLM agent runtime, I want to call search, find-similar, contents, answer, and research as tools over MCP, so that I can use the platform without writing a custom HTTP client.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose `search`, `find_similar`, `contents`, `answer`, and `research` as MCP tools, each declaring a JSON Schema for its input arguments and a JSON Schema for its response payload.
2. WHEN an MCP client invokes a tool with arguments that validate against the tool's input schema, THE MCP_Server SHALL forward the call to the corresponding subsystem (Retriever for `search` and `find_similar`, Search_Engine for `contents`, Answer_Engine for `answer`, Research_Agent for `research`) and SHALL return a response payload that validates against the tool's output schema.
3. IF an MCP tool call's arguments fail input schema validation, THEN THE MCP_Server SHALL reject the call with an MCP-standard validation error whose payload identifies the offending argument path and the failed schema constraint and SHALL NOT invoke the underlying subsystem.
4. THE MCP_Server SHALL authenticate every tool call using the same tenant-scoped API key mechanism as the REST API_Gateway and SHALL apply the same per-Tenant rate limits as the REST API_Gateway for the corresponding endpoints.
5. IF an MCP tool call presents an API key that is missing, invalid, expired, or revoked, THEN THE MCP_Server SHALL reject the call with an MCP-standard authentication error and SHALL NOT invoke the underlying subsystem.
6. IF an authenticated MCP tool call would cause the calling Tenant to exceed its per-Tenant rate limit for the corresponding endpoint, THEN THE MCP_Server SHALL reject the call with an MCP-standard rate-limit error indicating when the client may retry and SHALL NOT invoke the underlying subsystem.
7. IF the underlying subsystem returns an error or produces a response that fails the tool's output schema, THEN THE MCP_Server SHALL return an MCP-standard tool-execution error to the MCP client identifying the failure category and SHALL NOT return a partial or malformed payload.

### Requirement 13: Authentication and Tenant Isolation

**User Story:** As a tenant administrator, I want strong authentication and strict isolation between tenants, so that my data and traffic cannot be observed or affected by other customers.

#### Acceptance Criteria

1. WHEN the API_Gateway receives any request, THE Auth_Service SHALL authenticate the request using a tenant-scoped API key transmitted in the `Authorization: Bearer` header, complete authentication within 50 milliseconds at the 95th percentile, and resolve the request's owning Tenant before any business logic executes.
2. IF a request omits the `Authorization` header, presents a malformed bearer token, or presents an API key that is unknown, expired, or revoked, THEN THE API_Gateway SHALL reject the request with HTTP 401 and an `unauthenticated` error code, SHALL NOT invoke any downstream subsystem, and SHALL NOT include any tenant-scoped data in the response.
3. THE Auth_Service SHALL ensure that every read or write of session data, pipeline definitions, audit entries, research artifacts, and metering records is gated by an authorization check requiring the resource's `tenant_id` to equal the authenticated request's `tenant_id`, with cross-tenant attempts rejected as if the resource did not exist (HTTP 404 with the resource's standard not-found error code).
4. WHEN an administrator revokes an API key, THE Auth_Service SHALL reject all subsequent requests presenting that key with HTTP 401 and an `unauthenticated` error code within 60 seconds of revocation acceptance.
5. WHEN an administrator rotates an API key, THE Auth_Service SHALL accept both the previous key and the new key as valid during a configurable grace period in the integer range [1, 86400] seconds (defaulting to 3,600 seconds and capped at 24 hours), after which only the new key SHALL authenticate.
6. WHEN the Auth_Service rejects a request for any authentication failure listed above, THE Auth_Service SHALL emit an `auth_failure` entry to the Audit_Log capturing the failure reason code, the request identifier, the source IP, and a UTC timestamp, SHALL NOT log the presented bearer token value, and SHALL NOT correlate the entry to a Tenant when the key is unknown.

### Requirement 14: Rate Limiting and Usage Metering

**User Story:** As a platform operator, I want per-tenant rate limits and accurate usage metering, so that I can prevent abuse and bill customers fairly.

#### Acceptance Criteria

1. WHEN a Tenant exceeds its configured requests-per-minute limit on an endpoint, THE API_Gateway SHALL respond with HTTP 429, a `Retry-After` header in the integer range [1, 3600] seconds, a `rate_limited` error code, and the headers `X-RateLimit-Limit`, `X-RateLimit-Remaining` set to 0, and `X-RateLimit-Reset` as a Unix epoch second.
2. THE API_Gateway SHALL define a "billable request" as any HTTP 2xx response on `/v1/search`, `/v1/find_similar`, `/v1/contents`, `/v1/answer`, or `/v1/research`, and SHALL emit exactly one metering event per billable request capturing `tenant_id`, `endpoint`, `request_id`, `units_consumed`, and `timestamp` in ISO 8601 UTC.
3. WHEN the API_Gateway emits metering events, THE API_Gateway SHALL guarantee at-least-once delivery to the metering pipeline and SHALL ensure downstream deduplication by `request_id`.
4. WHILE a Tenant is within its rate limits on an endpoint, THE API_Gateway SHALL include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` (Unix epoch second) headers on every response from that endpoint.
5. IF the metering pipeline is unreachable when the API_Gateway attempts to publish a metering event, THEN THE API_Gateway SHALL persist the event to a durable local buffer, SHALL retry delivery with bounded backoff, SHALL NOT block or fail the corresponding API response, and SHALL emit an `metering_delivery_degraded` event to the Audit_Log when the local buffer reaches 80% of its configured capacity.

### Requirement 15: Audit Logging, PII Redaction, and Data Retention

**User Story:** As a security and compliance officer, I want every privileged action recorded and personal data handled per policy, so that the platform is defensible to auditors and customers.

#### Acceptance Criteria

1. WHEN any of the following occur, THE Audit_Log SHALL append an immutable entry within 5 seconds of the action capturing `actor`, `action`, `resource`, `timestamp` in ISO 8601 UTC, and `request_id` (a string of length between 16 and 64 code points): API key creation or revocation, pipeline create/update/delete, session create/delete, research job launch, data export, configuration change.
2. WHEN the API_Gateway receives a `query` field for `/v1/search`, `/v1/answer`, or `/v1/research`, THE PII_Redactor SHALL detect and redact, before any copy of the query is written to the Audit_Log or analytics pipeline, the pattern types `email_address` (RFC 5322 addr-spec), `phone_number` (E.164), `us_ssn`, `eu_national_id`, and `credit_card_number` (PAN with passing Luhn check).
3. WHEN a Tenant submits a data deletion request for a `session_id`, `research_job_id`, or the entire Tenant, THE Search_Engine SHALL acknowledge the request with HTTP 202 within 5 seconds, SHALL delete all associated session state, research artifacts, and metering records older than the legally required retention window within 30 days of acknowledgment, and SHALL emit a `deletion_completed` event to the Audit_Log capturing the request identifier and the count of records deleted.
4. THE Search_Engine SHALL retain Audit_Log entries for a configurable retention period in the integer range [365, 2,555] days (defaulting to 365), SHALL prevent in-place modification of any existing entry, and SHALL reject any write operation other than append.
5. IF a deletion request targets data that the platform is legally required to retain, THEN THE Search_Engine SHALL refuse the deletion for the affected records, SHALL include for each refused record an entry containing `record_id` and a `retention_required` reason code, and SHALL proceed with deletion of all remaining records.
6. IF the Audit_Log append fails for an action listed in criterion 1, THEN THE Search_Engine SHALL block completion of the privileged action, SHALL surface an `audit_log_unavailable` error to the caller, and SHALL retry the append until success or operator intervention.
7. IF a deletion request supplies a `session_id` or `research_job_id` that does not belong to the calling Tenant, THEN THE API_Gateway SHALL respond with HTTP 404 and a `resource_not_found` error code without disclosing whether the resource exists in another Tenant and SHALL NOT delete any records.

### Requirement 16: Python and TypeScript SDK Support

**User Story:** As a developer, I want idiomatic Python and TypeScript SDKs that mirror the REST API, so that I can integrate quickly without writing a transport layer.

#### Acceptance Criteria

1. THE Search_Engine SHALL distribute a Python SDK via the Python Package Index and a TypeScript SDK via the npm registry, each exposing type-annotated client methods for `search`, `find_similar`, `contents`, `answer`, `research`, `sessions`, and `pipelines`.
2. WHEN a streaming endpoint (`/v1/answer` with `stream: true`, or `GET /v1/research/{job_id}/events`) is called via either SDK, THE SDK SHALL expose results as an async iterator yielding the event types defined in the REST contract (`token`, `citation`, `step_started`, `step_completed`, `report_chunk`, `done`, `error`) and SHALL terminate iteration after yielding a `done` or `error` event.
3. WHEN the API_Gateway returns a non-2xx response, the network connection times out, the network connection fails, or the SDK fails to parse the response payload, THE SDK SHALL raise a typed exception whose fields, when available, include the HTTP status, error code, and request identifier returned by the API_Gateway.
4. THE Python SDK and the TypeScript SDK SHALL be generated from or validated against a single OpenAPI specification published at `/v1/openapi.json` such that every endpoint defined in that specification has a corresponding SDK method and every parameter type, response type, and error type matches the specification.
5. THE Python SDK and the TypeScript SDK SHALL transmit the configured tenant-scoped API key in the `Authorization: Bearer` header on every outbound request consistent with Requirement 13.
