## Overview

This MCP server targets Elasticsearch clusters that store microservice log data. The original flow streamed raw search hits into an LLM, which quickly exceeded model limits. The new design adds insight-focused tools so downstream callers can ask the server for targeted summaries and small, representative samples instead of large result sets.

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

- `summarize_logs`: produces aggregate counts (status, service, user), latency stats, slow-path highlights, and a handful of recent samples for context. Accepts an optional lookback window and field overrides.
- `log_trend`: returns a date histogram over the requested lookback period, including per-bucket status distributions and average latency (when numeric data is available).
- `sample_trace`: fetches a capped set of events for a specific trace ID, alongside span and status distributions.

All tools clamp query sizes, set short timeouts, and only emit the tailored structures needed by an LLM or planner, dramatically reducing token usage compared to streaming entire hit sets.

## Existing Utilities

The original tools (`list_indices`, `get_mapping`, `get_field_caps`, `sample_values`, `execute_search`, and optional `plan_query`) remain unchanged and can be combined with the new insight endpoints for richer investigative workflows.
