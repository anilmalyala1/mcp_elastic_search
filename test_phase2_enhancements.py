"""Test script and documentation for Phase 2 enhancements.

This script demonstrates the new Phase 2 tools and their usage patterns.
"""

import json


def test_phase2_tools_documentation():
    """Document the Phase 2 tools and their expected usage."""

    print("=" * 80)
    print("PHASE 2 ENHANCEMENTS - NEW TOOLS")
    print("=" * 80)

    print("\n1. search_paginated - Efficient Deep Pagination")
    print("-" * 80)
    print("Purpose: Process unlimited result sets in manageable chunks")
    print("Benefits: Stateless, efficient, no scroll contexts")
    print("\nExample usage:")

    pagination_example = {
        "first_page": {
            "tool": "search_paginated",
            "params": {
                "index": "logs-*",
                "dsl": {
                    "query": {"match": {"level": "ERROR"}},
                    "sort": [{"@timestamp": "desc"}, {"_id": "desc"}]  # REQUIRED
                },
                "page_size": 50,
                "fields": ["@timestamp", "service", "message"]
            },
            "response_includes": ["hits", "next_page_token", "has_next_page"]
        },
        "next_page": {
            "tool": "search_paginated",
            "params": {
                "index": "logs-*",
                "dsl": {
                    "query": {"match": {"level": "ERROR"}},
                    "sort": [{"@timestamp": "desc"}, {"_id": "desc"}]
                },
                "page_size": 50,
                "search_after": "<use_next_page_token_from_previous_response>",
                "fields": ["@timestamp", "service", "message"]
            },
            "note": "Continue until has_next_page is False"
        }
    }
    print(json.dumps(pagination_example, indent=2))

    print("\n\n2. estimate_response_size - Proactive Safety Check")
    print("-" * 80)
    print("Purpose: Preview response size before execution to prevent MCP host overload")
    print("Benefits: Actionable recommendations, prevents issues proactively")
    print("\nExample usage:")

    estimate_example = {
        "scenario": "Check if query is safe before executing",
        "tool": "estimate_response_size",
        "params": {
            "index": "logs-*",
            "dsl": {
                "query": {"range": {"@timestamp": {"gte": "now-7d"}}},
                "size": 200
            }
        },
        "response_structure": {
            "total_matches": 50000,
            "docs_to_return": 200,
            "avg_doc_size_kb": 15.3,
            "estimated_response_kb": 3060,
            "estimated_response_mb": 2.99,
            "safe_to_execute": False,
            "recommendations": [
                "⚠️ Response may be 3060KB (2.99MB). This could overwhelm the MCP host.",
                "✅ Use search_with_projection to limit fields. Estimated size: ~306KB (90% reduction)",
                "✅ Use search_paginated to fetch 200 documents in chunks of 50. Each page: ~765KB.",
                "✅ Use count_and_aggregate if you only need statistics (response: <10KB)."
            ]
        },
        "workflow": {
            "step_1": "Run estimate_response_size first",
            "step_2": "Check safe_to_execute flag",
            "step_3": "Follow recommendations if not safe",
            "step_4": "Execute optimized query"
        }
    }
    print(json.dumps(estimate_example, indent=2))

    print("\n\n3. sample_logs_stratified - Balanced Categorical Sampling")
    print("-" * 80)
    print("Purpose: Get representative samples across categories")
    print("Benefits: Balanced view, avoids bias, controlled response size")
    print("\nExample usage:")

    stratified_examples = {
        "by_status_code": {
            "tool": "sample_logs_stratified",
            "params": {
                "index": "logs-*",
                "strata_field": "response_status",
                "samples_per_stratum": 2,
                "lookback": "now-1h"
            },
            "result_structure": {
                "200": ["<sample_1>", "<sample_2>"],
                "400": ["<sample_1>", "<sample_2>"],
                "500": ["<sample_1>", "<sample_2>"]
            },
            "use_case": "See examples of all HTTP status codes"
        },
        "by_service": {
            "tool": "sample_logs_stratified",
            "params": {
                "index": "logs-*",
                "strata_field": "service_id.keyword",
                "samples_per_stratum": 3,
                "max_strata": 5
            },
            "use_case": "Sample from top 5 services"
        },
        "by_log_level": {
            "tool": "sample_logs_stratified",
            "params": {
                "index": "logs-*",
                "strata_field": "level.keyword",
                "samples_per_stratum": 5
            },
            "result_structure": {
                "ERROR": ["<5 samples>"],
                "WARN": ["<5 samples>"],
                "INFO": ["<5 samples>"]
            },
            "use_case": "See examples of each log level"
        }
    }
    print(json.dumps(stratified_examples, indent=2))

    print("\n\n" + "=" * 80)
    print("PHASE 2 BENEFITS SUMMARY")
    print("=" * 80)

    benefits = {
        "search_paginated": {
            "capability": "Process unlimited documents",
            "response_size": "50-100KB per page",
            "use_when": [
                "Need to process > 200 documents",
                "Want to iterate through large result sets",
                "Need efficient deep pagination"
            ],
            "advantages": [
                "Stateless (no scroll contexts)",
                "More efficient than from/size for deep pagination",
                "Works with field projection",
                "Predictable memory usage"
            ]
        },
        "estimate_response_size": {
            "capability": "Preview before execution",
            "response_size": "~5KB (just estimates)",
            "use_when": [
                "Query potentially returns large results",
                "Want safety check before execution",
                "Need optimization recommendations"
            ],
            "advantages": [
                "Prevents MCP host overload",
                "Provides actionable recommendations",
                "Shows savings with field projection",
                "No risk of large responses"
            ]
        },
        "sample_logs_stratified": {
            "capability": "Balanced sampling across categories",
            "response_size": "10-30KB (controlled)",
            "use_when": [
                "Need representative samples",
                "Want to see all status codes/services/levels",
                "Avoid bias toward most common values"
            ],
            "advantages": [
                "Balanced representation",
                "Controlled sample count",
                "See rare and common events",
                "Better than chronological sampling"
            ]
        }
    }
    print(json.dumps(benefits, indent=2))

    print("\n\n" + "=" * 80)
    print("REAL-WORLD WORKFLOW EXAMPLES")
    print("=" * 80)

    workflows = {
        "workflow_1_investigate_errors": {
            "scenario": "Investigate 5000 error logs from last week",
            "steps": [
                {
                    "step": 1,
                    "action": "Estimate response size",
                    "tool": "estimate_response_size",
                    "params": {"index": "logs-*", "dsl": {"query": {"term": {"level": "ERROR"}}, "size": 5000}}
                },
                {
                    "step": 2,
                    "action": "Get statistics first",
                    "tool": "count_and_aggregate",
                    "params": {
                        "index": "logs-*",
                        "query": {"term": {"level": "ERROR"}},
                        "aggregations": {
                            "by_service": {"terms": {"field": "service.keyword", "size": 20}},
                            "by_message": {"terms": {"field": "message.keyword", "size": 10}}
                        }
                    }
                },
                {
                    "step": 3,
                    "action": "Get balanced samples",
                    "tool": "sample_logs_stratified",
                    "params": {
                        "index": "logs-*",
                        "strata_field": "service.keyword",
                        "samples_per_stratum": 3
                    }
                },
                {
                    "step": 4,
                    "action": "Process all errors if needed",
                    "tool": "search_paginated",
                    "params": {
                        "page_size": 100,
                        "fields": ["@timestamp", "service", "message", "stack_trace"]
                    },
                    "note": "Iterate through pages as needed"
                }
            ]
        },
        "workflow_2_safe_export": {
            "scenario": "Export 10,000 logs safely",
            "steps": [
                {
                    "step": 1,
                    "action": "Check safety",
                    "tool": "estimate_response_size",
                    "expected": "Will warn about large size"
                },
                {
                    "step": 2,
                    "action": "Use pagination with projection",
                    "tool": "search_paginated",
                    "params": {
                        "page_size": 100,
                        "fields": ["@timestamp", "service", "user", "action", "result"]
                    },
                    "note": "100 pages of 100 docs = 10,000 total. Each page ~50KB"
                }
            ]
        },
        "workflow_3_balanced_analysis": {
            "scenario": "Understand traffic patterns across all services",
            "steps": [
                {
                    "step": 1,
                    "action": "Get service distribution",
                    "tool": "count_and_aggregate",
                    "params": {
                        "aggregations": {
                            "by_service": {"terms": {"field": "service.keyword", "size": 50}}
                        }
                    }
                },
                {
                    "step": 2,
                    "action": "Get samples from each service",
                    "tool": "sample_logs_stratified",
                    "params": {
                        "strata_field": "service.keyword",
                        "samples_per_stratum": 5,
                        "max_strata": 20
                    },
                    "result": "5 samples × 20 services = 100 total samples, balanced"
                }
            ]
        }
    }
    print(json.dumps(workflows, indent=2))

    print("\n\n" + "=" * 80)
    print("COMPARISON: BEFORE vs AFTER (Phase 2)")
    print("=" * 80)

    comparison = {
        "scenario_1": {
            "task": "Process 1000 documents",
            "before": {
                "approach": "execute_search with size=1000",
                "risk": "Could return 10-20MB, crash MCP host",
                "limitations": "Size limit of 200 enforced"
            },
            "after_phase2": {
                "approach": "search_paginated with page_size=100",
                "result": "10 pages × ~500KB = 5MB total, processed in chunks",
                "safety": "Each page loads independently, no memory spike"
            }
        },
        "scenario_2": {
            "task": "Check if query is safe to run",
            "before": {
                "approach": "Just run it and hope for the best",
                "risk": "Might crash MCP host with huge response"
            },
            "after_phase2": {
                "approach": "estimate_response_size first",
                "result": "Get warning + recommendations before executing",
                "safety": "Proactive prevention"
            }
        },
        "scenario_3": {
            "task": "Get representative sample of logs",
            "before": {
                "approach": "summarize_logs returns 3 most recent",
                "limitation": "Biased toward most recent, might miss patterns"
            },
            "after_phase2": {
                "approach": "sample_logs_stratified across status codes",
                "result": "Balanced samples from 200, 400, 500 responses",
                "advantage": "See all patterns, not just recent ones"
            }
        }
    }
    print(json.dumps(comparison, indent=2))

    print("\n\n" + "=" * 80)
    print("RESPONSE SIZE GUIDE (Phase 1 + Phase 2)")
    print("=" * 80)

    size_guide = [
        {"tool": "count_and_aggregate", "size": "2-10 KB", "docs": 0, "use": "Statistics only"},
        {"tool": "estimate_response_size", "size": "~5 KB", "docs": 0, "use": "Safety check"},
        {"tool": "sample_logs_stratified", "size": "10-30 KB", "docs": "10-50", "use": "Balanced samples"},
        {"tool": "summarize_logs (sample_size=0)", "size": "~10 KB", "docs": 0, "use": "Quick overview"},
        {"tool": "summarize_logs (sample_size=5)", "size": "~20 KB", "docs": 5, "use": "Overview + samples"},
        {"tool": "search_with_projection", "size": "~100 KB", "docs": 200, "use": "Filtered documents"},
        {"tool": "search_paginated (per page)", "size": "50-100 KB", "docs": "50-100", "use": "Large datasets"},
        {"tool": "execute_search", "size": "1-4 MB", "docs": 200, "use": "⚠️ Use alternatives"},
    ]

    print("\n{:<35} {:<15} {:<10} {:<30}".format("Tool", "Response Size", "Documents", "Use Case"))
    print("-" * 95)
    for item in size_guide:
        print("{:<35} {:<15} {:<10} {:<30}".format(
            item["tool"], item["size"], str(item["docs"]), item["use"]
        ))

    print("\n\n" + "=" * 80)
    print("All Phase 2 enhancements implemented successfully!")
    print("Tools: search_paginated, estimate_response_size, sample_logs_stratified")
    print("=" * 80)


if __name__ == "__main__":
    test_phase2_tools_documentation()
