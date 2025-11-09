"""Utility script to push transaction logs into an Elasticsearch index.

This script can either ingest records from a newline-delimited JSON file or
generate synthetic transactions (useful for load testing with ~100K docs).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from elasticsearch import Elasticsearch, exceptions as es_exceptions, helpers

from es_client import create_elasticsearch_client

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_INDEX = os.getenv("TRANSACTIONS_INDEX", "transactions")
DEFAULT_COUNT = 100_000
DEFAULT_BATCH_SIZE = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk ingest transaction logs into Elasticsearch.",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Target index name (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to newline-delimited JSON file (.jsonl/.ndjson) with transaction records.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of synthetic records to generate when --file is not provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents to send per bulk request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for synthetic data generation.",
    )
    return parser.parse_args()


def ensure_index(client: Elasticsearch, index: str) -> None:
    """Create the target index with a reasonable mapping if it is missing."""

    try:
        exists = client.indices.exists(index=index)
    except es_exceptions.ElasticsearchException as exc:  # pragma: no cover - network error
        raise RuntimeError(f"Failed to check if index '{index}' exists: {exc}") from exc

    if exists:
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "transaction_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "amount": {"type": "double"},
                "currency": {"type": "keyword"},
                "status": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "merchant": {"type": "keyword"},
                "merchant_category": {"type": "keyword"},
                "payment_method": {"type": "keyword"},
                "country": {"type": "keyword"},
                "city": {"type": "keyword"},
                "device_type": {"type": "keyword"},
                "ip_address": {"type": "ip"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }

    try:
        client.indices.create(index=index, body=mapping)
        print(f"Created index '{index}' with default mapping.")
    except es_exceptions.ElasticsearchException as exc:  # pragma: no cover - network error
        raise RuntimeError(f"Index creation failed for '{index}': {exc}") from exc


def iter_file_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON documents from a newline-delimited JSON file or JSON array."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            first_non_ws = handle.read(1)
            handle.seek(0)
            if first_non_ws == "[":
                data = json.load(handle)
                if not isinstance(data, list):
                    raise ValueError("JSON root must be an array when file starts with '['.")
                for entry in data:
                    if isinstance(entry, dict):
                        yield entry
            else:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        yield obj
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc


def generate_transactions(count: int, seed: int | None) -> Iterator[Dict[str, Any]]:
    """Generate synthetic transaction documents for testing."""

    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    currencies = ["USD", "EUR", "GBP", "INR", "JPY"]
    statuses = ["completed", "pending", "failed", "refunded"]
    merchants = [
        "Acme Corp",
        "Globex",
        "Soylent",
        "Initech",
        "Hooli",
        "Stark Industries",
        "Wayne Enterprises",
    ]
    categories = [
        "electronics",
        "groceries",
        "entertainment",
        "travel",
        "utilities",
        "fashion",
    ]
    payment_methods = ["card", "wallet", "bank_transfer", "upi"]
    countries = ["US", "GB", "IN", "DE", "JP", "AU"]
    devices = ["web", "ios", "android"]

    for _ in range(count):
        transaction_id = str(uuid.uuid4())
        record = {
            "transaction_id": transaction_id,
            "user_id": f"user_{rng.randint(1, 1_000_000)}",
            "amount": round(rng.uniform(1.0, 2_500.0), 2),
            "currency": rng.choice(currencies),
            "status": rng.choices(statuses, weights=[0.85, 0.1, 0.03, 0.02], k=1)[0],
            "timestamp": (now - timedelta(seconds=rng.randint(0, 3600 * 24 * 30))).isoformat(),
            "merchant": rng.choice(merchants),
            "merchant_category": rng.choice(categories),
            "payment_method": rng.choice(payment_methods),
            "country": rng.choice(countries),
            "city": f"city_{rng.randint(1, 5000)}",
            "device_type": rng.choice(devices),
            "ip_address": ".".join(str(rng.randint(0, 255)) for _ in range(4)),
            "metadata": {
                "session_id": str(uuid.uuid4()),
                "promotion_applied": rng.random() < 0.1,
            },
        }
        yield record


def build_actions(records: Iterable[Dict[str, Any]], index: str) -> Iterator[Dict[str, Any]]:
    """Wrap records for helpers.bulk while ensuring IDs are present."""

    for record in records:
        if not isinstance(record, dict):
            continue
        doc = dict(record)
        doc_id = doc.get("transaction_id") or str(uuid.uuid4())
        doc["transaction_id"] = doc_id
        yield {
            "_index": index,
            "_id": doc_id,
            "_source": doc,
        }


def bulk_ingest(
    client: Elasticsearch,
    index: str,
    records: Iterable[Dict[str, Any]],
    batch_size: int,
) -> int:
    """Send documents in batches using the helpers.bulk API."""

    actions = build_actions(records, index)
    try:
        success_count, errors = helpers.bulk(
            client,
            actions,
            chunk_size=max(1, batch_size),
            request_timeout=120,
        )
    except es_exceptions.ElasticsearchException as exc:  # pragma: no cover - network error
        raise RuntimeError(f"Bulk ingest failed: {exc}") from exc

    if errors:
        print(f"Encountered {len(errors)} errors during ingestion.", file=sys.stderr)
    return success_count


def main() -> None:
    if load_dotenv:
        load_dotenv()

    args = parse_args()

    es_url = os.getenv("ES_URL", "http://localhost:9200")
    es_api_key = os.getenv("ES_API_KEY")

    client = create_elasticsearch_client(es_url, es_api_key)
    ensure_index(client, args.index)

    if args.file:
        records_iterable = iter_file_records(args.file)
    else:
        records_iterable = generate_transactions(args.count, args.seed)

    ingested = bulk_ingest(client, args.index, records_iterable, args.batch_size)
    print(f"Ingested {ingested} documents into '{args.index}'.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - entry-point safety
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
