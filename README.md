## Overview

This project provides an **MCP (Multi-Cloud Platform) server** designed to interact with Elasticsearch clusters that store microservice log data. This server exposes a comprehensive set of tools, allowing various clients and users to ask for targeted summaries and small, representative samples of log data, rather than processing large raw result sets.

The server's design focuses on efficiency and reducing the burden on downstream systems, particularly when integrating with Language Models (LLMs). The original approach of streaming raw search hits often exceeded model token limits. This new design enhances insight-focused tools to provide curated, actionable data, optimizing for both performance and LLM integration.

## Getting Started

To run the MCP Elastic server, navigate to the project directory and execute the `mcp_elastic.py` script. The server will start and expose its functionality via the `streamable-http` transport.

```bash
python3 mcp_elastic.py
```

## Configuration

The MCP Elastic server can be configured using the following environment variables:

| Environment Variable  | Description                                                         | Default Value             |
|-----------------------|---------------------------------------------------------------------|---------------------------|
| `ES_URL`              | URL of the Elasticsearch cluster.                                   | `http://localhost:9200`   |
| `ES_API_KEY`          | API key for Elasticsearch authentication.                           | `None`                    |
| `ENABLE_PLANNER`      | Enables or disables the LLM-powered query planner.                  | `true`                    |
| `OPENAI_API_KEY`      | OpenAI API key, required if `ENABLE_PLANNER` is `true`.             | `None`                    |
| `DEFAULT_TIME_FIELD`  | The default field used for time-based queries.                      | `@timestamp`              |
| `SEARCH_SIZE_LIMIT`   | Maximum number of documents to return in a single search request.   | `200`                     |
| `SEARCH_TIMEOUT_MS`   | Timeout for Elasticsearch search requests in milliseconds.          | `5000`                    |

## Default Log Shape

The helper utilities assume logs expose the following canonical fields. Use `field_overrides` on the insight-focused tools when your mapping uses different names.

| Logical field    | Default mapping   | Notes                                  |
| ---------------- | ----------------- | -------------------------------------- |
| `trace_id`       | `trace_id`        | Used to correlate requests end-to-end. |
| `span_id`        | `span_id`         | Span-level correlation.                |
| `message`        | `log_message`     | Human-readable log text.               |
| `service`        | `service_id`      | Service or component identifier.       |
| `user`           | `userid`          | Authenticated user or actor.           |
| `status`         | `response_status` | HTTP/code-style response indicator.    |
| `response_time`  | `response_time`   | Duration in milliseconds.              |
| `path`           | `path`            | Endpoint or route.                     |
| `time`           | `@timestamp`      | Used for time filtering.               |

When a mapped field also exposes a `.keyword` subfield, the server automatically prefers it for aggregations while still reading the structured value from `_source` for samples.

## Insight Tools

- `summarize_logs`: produces aggregate counts (status, service, user), latency stats, slow-path highlights, and configurable log samples. Now supports:
  - Configurable sample size (0-10 samples)
  - Multiple sampling strategies (recent, random, or none)
  - Configurable message truncation to control response size
  - Use `sample_size=0` or `sample_strategy="none"` for aggregations-only results
- `log_trend`: returns a date histogram over the requested lookback period, including per-bucket status distributions and average latency (when numeric data is available).
- `sample_trace`: fetches a capped set of events for a specific trace ID, alongside span and status distributions.

All tools clamp query sizes, set short timeouts, and only emit the tailored structures needed by an LLM or planner, dramatically reducing token usage compared to streaming entire hit sets.

## Large Dataset Tools (Phase 1 Enhancements)

To prevent overwhelming MCP hosts with huge responses, the following tools have been added:

- `search_with_projection`: Execute searches with field projection to return only specific fields. Can reduce response size by 90-95% compared to returning full documents. Perfect for when you need documents but only specific fields.

- `count_and_aggregate`: Execute aggregation-only queries without retrieving any documents. Returns only statistics and aggregations, making it safe to analyze millions of records. Response sizes are typically just a few KB regardless of dataset size.

## Advanced Data Handling Tools (Phase 2 Enhancements)

Building on Phase 1, these tools add pagination, proactive safety checks, and balanced sampling:

- `search_paginated`: Process unlimited result sets in manageable chunks using efficient search_after pagination. Stateless and more efficient than from/size for deep pagination. Supports field projection for optimal performance.

- `estimate_response_size`: Preview estimated response size before executing a query. Provides actionable recommendations for optimization (use projection, pagination, or aggregation). Prevents MCP host overload proactively.

- `sample_logs_stratified`: Retrieve balanced samples across categories (e.g., one sample from each HTTP status code, or from each service). Provides better representation than chronological sampling, ideal for understanding diverse log patterns.

These enhancements enable efficient analysis of large log datasets without memory issues or performance degradation.

## Planner Tools

These tools leverage Language Models (LLMs) to provide more intelligent querying capabilities.

- `plan_query`: Translates a natural language question into an Elasticsearch Query DSL. This tool uses an LLM (if configured) or local heuristics to convert a user's question (e.g., "show me login errors from the last hour") into a valid Elasticsearch DSL query. The result includes the generated DSL, the target indices, and the planner's confidence and assumptions. This tool is only available if `ENABLE_PLANNER` is set to `true` and `OPENAI_API_KEY` is configured.

## Existing Utilities

The original tools (`list_indices`, `get_mapping`, `get_field_caps`, `sample_values`, `execute_search`, and optional `plan_query`) remain unchanged and can be combined with the new insight endpoints for richer investigative workflows.
