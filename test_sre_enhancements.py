
import unittest
from unittest.mock import MagicMock, patch
import json
from mcp_elastic import (
    get_cluster_health,
    find_traces_by_user,
    analyze_error_patterns,
    get_metric_statistics,
    get_top_values,
    sample_trace,
    get_es_client,
    server
)

class TestSREEnhancements(unittest.TestCase):

    @patch('mcp_elastic.get_es_client')
    def test_get_cluster_health(self, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.cluster.health.return_value = {
            "status": "green",
            "cluster_name": "test-cluster",
            "number_of_nodes": 3,
            "active_shards": 10,
            "unassigned_shards": 0
        }

        # Execute
        result = get_cluster_health()

        # Verify
        self.assertEqual(result["status"], "green")
        self.assertEqual(result["number_of_nodes"], 3)
        mock_client.cluster.health.assert_called_once()

    @patch('mcp_elastic.get_es_client')
    @patch('mcp_elastic.extract_field_inventory_for_indices')
    def test_find_traces_by_user(self, mock_extract, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_extract.return_value = [
            {"name": "user.id", "type": "keyword"},
            {"name": "trace.id", "type": "keyword"},
            {"name": "@timestamp", "type": "date"},
            {"name": "service.name", "type": "keyword"}
        ]
        
        mock_response = {
            "aggregations": {
                "traces": {
                    "buckets": [
                        {
                            "key": "trace-123",
                            "doc_count": 5,
                            "latest_time": {"value_as_string": "2023-10-27T10:00:00Z"},
                            "services": {"buckets": [{"key": "frontend"}, {"key": "backend"}]}
                        }
                    ]
                }
            }
        }
        mock_client.search.return_value = mock_response

        # Execute
        result = find_traces_by_user(
            index="logs-*",
            user_id="user-123",
            field_overrides={"user": "user.id", "trace_id": "trace.id", "service": "service.name"}
        )

        # Verify
        self.assertEqual(result["user_id"], "user-123")
        self.assertEqual(len(result["traces"]), 1)
        self.assertEqual(result["traces"][0]["trace_id"], "trace-123")
        self.assertEqual(result["traces"][0]["services"], ["frontend", "backend"])

    @patch('mcp_elastic.get_es_client')
    @patch('mcp_elastic.extract_field_inventory_for_indices')
    def test_analyze_error_patterns(self, mock_extract, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_extract.return_value = [
            {"name": "message", "type": "text", "hasKeyword": True},
            {"name": "status", "type": "integer", "isNumeric": True},
            {"name": "@timestamp", "type": "date"},
            {"name": "service.name", "type": "keyword"}
        ]

        mock_response = {
            "aggregations": {
                "top_errors": {
                    "buckets": [
                        {
                            "key": "Connection timeout",
                            "doc_count": 50,
                            "sample_services": {"buckets": [{"key": "payment-service"}]}
                        }
                    ]
                }
            }
        }
        mock_client.search.return_value = mock_response

        # Execute
        result = analyze_error_patterns(
            index="logs-*",
            field_overrides={"message": "message", "status": "status", "service": "service.name"}
        )

        # Verify
        self.assertEqual(len(result["patterns"]), 1)
        self.assertEqual(result["patterns"][0]["error_message"], "Connection timeout")
        self.assertEqual(result["patterns"][0]["count"], 50)
        self.assertEqual(result["patterns"][0]["affected_services"], ["payment-service"])

    @patch('mcp_elastic.get_es_client')
    @patch('mcp_elastic.extract_field_inventory_for_indices')
    def test_get_metric_statistics(self, mock_extract, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_extract.return_value = [
            {"name": "db.connections", "type": "long", "isNumeric": True},
            {"name": "@timestamp", "type": "date"}
        ]

        # Test scalar metric
        mock_client.search.return_value = {
            "aggregations": {
                "overall_metric": {"value": 42.5}
            }
        }

        result = get_metric_statistics(
            index="metrics-*",
            field="db.connections",
            metric="avg"
        )
        self.assertEqual(result["value"], 42.5)
        self.assertEqual(result["metric"], "avg")

        # Test trend metric
        mock_client.search.return_value = {
            "aggregations": {
                "trend": {
                    "buckets": [
                        {
                            "key_as_string": "2023-10-27T10:00:00Z",
                            "metric_value": {"value": 10}
                        },
                        {
                            "key_as_string": "2023-10-27T10:05:00Z",
                            "metric_value": {"value": 20}
                        }
                    ]
                }
            }
        }

        result = get_metric_statistics(
            index="metrics-*",
            field="db.connections",
            metric="max",
            interval="5m"
        )
        self.assertEqual(len(result["trend"]), 2)
        self.assertEqual(result["trend"][0]["value"], 10)
        self.assertEqual(result["trend"][1]["value"], 20)

    @patch('mcp_elastic.get_es_client')
    @patch('mcp_elastic.extract_field_inventory_for_indices')
    def test_get_top_values(self, mock_extract, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_extract.return_value = [
            {"name": "host.name", "type": "keyword", "hasKeyword": True},
            {"name": "@timestamp", "type": "date"}
        ]

        mock_client.search.return_value = {
            "aggregations": {
                "top_values": {
                    "buckets": [
                        {"key": "host-1", "doc_count": 100},
                        {"key": "host-2", "doc_count": 50}
                    ]
                }
            }
        }

        result = get_top_values(
            index="metrics-*",
            field="host.name",
            size=5
        )
        
        self.assertEqual(len(result["top_values"]), 2)
        self.assertEqual(result["top_values"][0]["value"], "host-1")
        self.assertEqual(result["top_values"][0]["count"], 100)

    @patch('mcp_elastic.get_es_client')
    @patch('mcp_elastic.extract_field_inventory_for_indices')
    def test_sample_trace_numeric_status(self, mock_extract, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_extract.return_value = [
            {"name": "http.response.status_code", "type": "long", "isNumeric": True},
            {"name": "trace.id", "type": "keyword"},
            {"name": "@timestamp", "type": "date"}
        ]

        # Mock search response
        mock_client.search.return_value = {
            "hits": {"total": {"value": 1}, "hits": []},
            "aggregations": {
                "status_counts": {
                    "buckets": [{"key": 200, "doc_count": 1}]
                }
            }
        }

        # Execute
        sample_trace(
            index="logs-*",
            trace_id="trace-123",
            field_overrides={"status": "http.response.status_code"}
        )

        # Verify call args
        call_args = mock_client.search.call_args[1]
        status_agg = call_args["body"]["aggs"]["status_counts"]["terms"]
        self.assertEqual(status_agg["field"], "http.response.status_code")
        self.assertEqual(status_agg["missing"], -1)

if __name__ == '__main__':
    unittest.main()
