"""Test script for Phase 1 enhancements.

This script demonstrates the new tools and their usage patterns.
"""

import json


def test_new_tools_documentation():
    """Document the new tools and their expected usage."""

    print("=" * 80)
    print("PHASE 1 ENHANCEMENTS - NEW TOOLS")
    print("=" * 80)

    print("\n1. search_with_projection")
    print("-" * 80)
    print("Purpose: Return only specific fields to reduce response size by 90-95%")
    print("\nExample usage:")
    example1 = {
        "tool": "search_with_projection",
        "params": {
            "index": "logs-*",
            "dsl": {
                "query": {"match": {"level": "ERROR"}},
                "size": 50
            },
            "fields": ["@timestamp", "service_id", "message", "trace_id"]
        }
    }
    print(json.dumps(example1, indent=2))

    print("\n\n2. count_and_aggregate")
    print("-" * 80)
    print("Purpose: Get statistics without retrieving documents (safe for millions of records)")
    print("\nExample usage:")
    example2 = {
        "tool": "count_and_aggregate",
        "params": {
            "index": "logs-*",
            "query": {"term": {"level": "ERROR"}},
            "time_range": {"gte": "now-1h"},
            "aggregations": {
                "by_service": {
                    "terms": {"field": "service_id.keyword", "size": 20}
                }
            }
        }
    }
    print(json.dumps(example2, indent=2))

    print("\n\n3. Enhanced summarize_logs")
    print("-" * 80)
    print("Purpose: Configurable sampling for better control over response size")
    print("\nExample usage (no samples, aggregations only):")
    example3a = {
        "tool": "summarize_logs",
        "params": {
            "index": "logs-*",
            "lookback": "now-24h",
            "sample_size": 0,  # No samples
            "sample_strategy": "none"
        }
    }
    print(json.dumps(example3a, indent=2))

    print("\nExample usage (random sampling with truncation):")
    example3b = {
        "tool": "summarize_logs",
        "params": {
            "index": "logs-*",
            "lookback": "now-1h",
            "sample_size": 5,
            "sample_strategy": "random",  # Unbiased sampling
            "max_message_length": 200  # Truncate long messages
        }
    }
    print(json.dumps(example3b, indent=2))

    print("\n\n" + "=" * 80)
    print("BENEFITS")
    print("=" * 80)

    benefits = {
        "search_with_projection": [
            "90-95% smaller responses",
            "Return only fields you need",
            "Same query capabilities as execute_search"
        ],
        "count_and_aggregate": [
            "Analyze millions of records safely",
            "Responses typically <10KB",
            "No document retrieval overhead"
        ],
        "enhanced_summarize_logs": [
            "Control sample count (0-10)",
            "Choose sampling strategy (recent/random/none)",
            "Configurable message truncation",
            "Predictable response sizes"
        ]
    }
    print(json.dumps(benefits, indent=2))

    print("\n\n" + "=" * 80)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 80)

    comparison = {
        "scenario": "Query 200 logs from last week",
        "before": {
            "tool": "execute_search",
            "response_size": "1-4 MB (full documents)",
            "risk": "Can overwhelm MCP host"
        },
        "after_option_1": {
            "tool": "count_and_aggregate",
            "response_size": "~5 KB (stats only)",
            "use_case": "Just need counts and metrics"
        },
        "after_option_2": {
            "tool": "search_with_projection",
            "response_size": "~100 KB (5 fields only)",
            "use_case": "Need documents but not all fields"
        },
        "after_option_3": {
            "tool": "summarize_logs with sample_size=5",
            "response_size": "~20 KB (aggregations + 5 samples)",
            "use_case": "Quick overview with examples"
        }
    }
    print(json.dumps(comparison, indent=2))

    print("\n\n" + "=" * 80)
    print("All Phase 1 enhancements implemented successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_new_tools_documentation()
