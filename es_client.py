"""Utilities for creating Elasticsearch clients."""

from typing import Any, Optional

from elasticsearch import Elasticsearch


def create_elasticsearch_client(
    es_url: str,
    api_key: Optional[str] = None,
    **client_kwargs: Any,
) -> Elasticsearch:
    """Create an Elasticsearch client using the provided configuration."""

    params = dict(client_kwargs)
    if api_key:
        params["api_key"] = api_key
    return Elasticsearch(es_url, **params)
