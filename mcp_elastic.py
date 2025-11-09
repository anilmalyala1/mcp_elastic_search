"""Simple MCP server for Elasticsearch."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from elasticsearch import Elasticsearch, exceptions as es_exceptions
from jsonschema import Draft7Validator, ValidationError

from es_client import create_elasticsearch_client
from llm_provider import PlannerLLMClient
from dotenv import load_dotenv,find_dotenv
from mcp.server.fastmcp import FastMCP as MCPServer
load_dotenv(find_dotenv())





# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ES_API_KEY")
ENABLE_PLANNER = os.getenv("ENABLE_PLANNER", "true").strip().lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_TIME_FIELD = os.getenv("DEFAULT_TIME_FIELD", "@timestamp")
SEARCH_SIZE_LIMIT = int(os.getenv("SEARCH_SIZE_LIMIT", "200"))
SEARCH_TIMEOUT_MS = int(os.getenv("SEARCH_TIMEOUT_MS", "5000"))

# -----------------------------------------------------------------------------
# Server initialization
# -----------------------------------------------------------------------------

server = MCPServer("mcp-elastic")
_es_client: Optional[Elasticsearch] = None
_planner_client: Optional[PlannerLLMClient] = None

# -----------------------------------------------------------------------------
# Simple TTL cache helper
# -----------------------------------------------------------------------------


class SimpleTTLCache:
    """A very small TTL cache using a dictionary."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        expires_at = time.time() + self.ttl_seconds
        self._store[key] = (expires_at, value)

    def clear(self) -> None:
        self._store.clear()


mappings_cache = SimpleTTLCache(ttl_seconds=600)
field_caps_cache = SimpleTTLCache(ttl_seconds=600)


def get_planner_client() -> Optional[PlannerLLMClient]:
    """Return a shared planner client when planner support is enabled."""

    if not ENABLE_PLANNER:
        return None
    global _planner_client
    if _planner_client is None:
        _planner_client = PlannerLLMClient(OPENAI_API_KEY)
    return _planner_client

# -----------------------------------------------------------------------------
# Elasticsearch client factory
# -----------------------------------------------------------------------------


def get_es_client() -> Elasticsearch:
    """Create or reuse a singleton Elasticsearch client."""

    global _es_client
    if _es_client is None:
        _es_client = create_elasticsearch_client(ES_URL, ES_API_KEY)
    return _es_client


# -----------------------------------------------------------------------------
# Mapping helpers
# -----------------------------------------------------------------------------


def is_numeric_type(field_type: Optional[str]) -> bool:
    """Return True if the mapping type is numeric."""

    numeric_types = {
        "integer",
        "long",
        "short",
        "byte",
        "double",
        "float",
        "half_float",
        "scaled_float",
        "unsigned_long",
    }
    return field_type in numeric_types


def flatten_mappings(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten mapping properties into a list of field descriptors."""

    fields: List[Dict[str, Any]] = []

    def walk_properties(
        properties: Dict[str, Any],
        parent: str,
        nested_path: Optional[str],
    ) -> None:
        for field_name, field_data in properties.items():
            full_name = field_name if not parent else f"{parent}.{field_name}"
            field_type = field_data.get("type")

            current_nested = nested_path
            if field_type == "nested":
                current_nested = full_name
            elif field_data.get("include_in_parent"):
                current_nested = nested_path

            fields_obj = field_data.get("fields", {})
            has_keyword = field_type == "keyword"
            if not has_keyword:
                has_keyword = "keyword" in fields_obj

            field_entry = {
                "name": full_name,
                "type": field_type,
                "nestedPath": current_nested,
                "hasKeyword": has_keyword,
                "isText": field_type == "text",
                "isDate": field_type in {"date", "date_nanos"},
                "isNumeric": is_numeric_type(field_type),
            }
            fields.append(field_entry)

            sub_properties = field_data.get("properties")
            if sub_properties:
                walk_properties(sub_properties, full_name, current_nested)

    mapping_properties = mapping.get("properties", {})
    walk_properties(mapping_properties, "", None)
    return fields


# -----------------------------------------------------------------------------
# Field capabilities helper
# -----------------------------------------------------------------------------


def summarize_field_caps(raw_caps: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw field caps into a friendly dictionary."""

    summary: Dict[str, Any] = {}
    fields_section = raw_caps.get("fields", {})
    for field_name, type_info in fields_section.items():
        types_list: List[str] = []
        searchable = False
        aggregatable = False
        for type_name, info in type_info.items():
            types_list.append(type_name)
            if info.get("searchable"):
                searchable = True
            if info.get("aggregatable"):
                aggregatable = True
        summary[field_name] = {
            "types": types_list,
            "searchable": searchable,
            "aggregatable": aggregatable,
        }
    return summary


# -----------------------------------------------------------------------------
# DSL validation
# -----------------------------------------------------------------------------


def build_dsl_schema() -> Dict[str, Any]:
    """Create the JSON schema used to validate Elasticsearch DSL payloads."""

    return {
        "type": "object",
        "properties": {
            "query": {"type": "object"},
            "aggs": {"type": "object"},
            "size": {"type": "integer", "minimum": 0, "maximum": SEARCH_SIZE_LIMIT},
            "from": {"type": "integer", "minimum": 0},
            "sort": {"type": ["array", "object"]},
            "track_total_hits": {"type": ["boolean", "integer"]},
            "timeout": {"type": "string"},
        },
        "additionalProperties": False,
    }


DSL_SCHEMA = build_dsl_schema()
DSL_VALIDATOR = Draft7Validator(DSL_SCHEMA)


FORBIDDEN_KEYS = {"script", "script_score", "rescore", "highlight", "pit", "search_after"}


def check_forbidden_content(value: Any) -> bool:
    """Return True if the value contains forbidden query parts."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in FORBIDDEN_KEYS:
                return True
            if check_forbidden_content(nested_value):
                return True
    elif isinstance(value, list):
        for item in value:
            if check_forbidden_content(item):
                return True
    return False


def validate_and_prepare_dsl(dsl: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DSL and apply guardrails such as size clamping."""

    try:
        DSL_VALIDATOR.validate(dsl)
    except ValidationError as exc:
        raise ValueError(f"Invalid DSL: {exc.message}") from exc

    if check_forbidden_content(dsl):
        raise ValueError("DSL contains forbidden keys such as script or pit.")

    prepared = dict(dsl)

    if "size" in prepared:
        if prepared["size"] > SEARCH_SIZE_LIMIT:
            prepared["size"] = SEARCH_SIZE_LIMIT
    else:
        prepared["size"] = min(10, SEARCH_SIZE_LIMIT)

    if "timeout" not in prepared:
        prepared["timeout"] = f"{SEARCH_TIMEOUT_MS}ms"

    track_hits = prepared.get("track_total_hits")
    if track_hits is True:
        prepared["track_total_hits"] = 10000

    return prepared


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def parse_byte_size(text: Optional[str]) -> int:
    """Parse _cat byte sizes into integers."""

    if not text:
        return 0
    match = re.match(r"([0-9.]+)([a-zA-Z]*)", text)
    if not match:
        return 0
    value = float(match.group(1))
    suffix = match.group(2).lower()
    multipliers = {
        "": 1,
        "b": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4,
        "pb": 1024 ** 5,
    }
    multiplier = multipliers.get(suffix, 1)
    return int(value * multiplier)


def extract_field_inventory(indices: List[str]) -> List[Dict[str, Any]]:
    """Fetch mappings for the first provided index to describe fields."""

    if not indices:
        return []
    first_index = indices[0]
    cache_key = f"mapping:{first_index}"
    cached = mappings_cache.get(cache_key)
    if cached is not None:
        return cached
    client = get_es_client()
    try:
        mapping_response = client.indices.get_mapping(index=first_index)
    except es_exceptions.ElasticsearchException:
        return []
    index_data = mapping_response.get(first_index, {})
    field_list = flatten_mappings(index_data.get("mappings", {}))
    mappings_cache.set(cache_key, field_list)
    return field_list


# -----------------------------------------------------------------------------
# Planner helpers (optional LLM integration)
# -----------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = (
    "You are an assistant that converts natural language questions into Elasticsearch DSL. "
    "Pick indices by semantics. Map NL entities to fields using the provided schema. "
    "Use .keyword for exact values; match or match_phrase for free text. "
    "Wrap nested queries when a field lives under a nested path. "
    "Use range on the time field for date shorthands like today or last week. "
    "Output only strict JSON with keys: indices, dsl, confidence, assumptions, alternatives.\n\n"
    "Examples:\n"
    "User: show login errors today\n"
    "Assistant: {\"indices\": [\"logs-*\"], \"dsl\": {\"size\": 25, \"query\": {\"bool\": {\"must\": [{\"match\": {\"message\": \"login error\"}}], \"filter\": [{\"range\": {\"@timestamp\": {\"gte\": \"now/d\"}}}]}}}, \"confidence\": 0.7, \"assumptions\": [\"message holds log text\"], \"alternatives\": []}\n"
    "User: top 5 services by error count last week\n"
    "Assistant: {\"indices\": [\"logs-*\"], \"dsl\": {\"size\": 0, \"query\": {\"bool\": {\"filter\": [{\"range\": {\"@timestamp\": {\"gte\": \"now-7d\", \"lte\": \"now\"}}}]}}, \"aggs\": {\"top_services\": {\"terms\": {\"field\": \"service.keyword\", \"size\": 5}}}}, \"confidence\": 0.75, \"assumptions\": [\"service.keyword exists\"], \"alternatives\": []}\n"
    "User: request volume per hour for checkout last 24 hours\n"
    "Assistant: {\"indices\": [\"orders-*\"], \"dsl\": {\"size\": 0, \"query\": {\"bool\": {\"must\": [{\"match\": {\"endpoint\": \"checkout\"}}], \"filter\": [{\"range\": {\"@timestamp\": {\"gte\": \"now-24h\", \"lte\": \"now\"}}}]}}, \"aggs\": {\"per_hour\": {\"date_histogram\": {\"field\": \"@timestamp\", \"fixed_interval\": \"1h\"}}}}, \"confidence\": 0.8, \"assumptions\": [\"endpoint holds route name\"], \"alternatives\": []}\n"
)


def call_llm_planner(
    nl: str,
    indices: Optional[List[str]],
    field_inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Call an LLM planner if one is configured."""

    planner = get_planner_client()
    if not planner:
        return None

    payload = {
        "question": nl,
        "indices": indices or [],
        "fields": field_inventory,
        "defaults": {
            "time_field": DEFAULT_TIME_FIELD,
            "size_limit": SEARCH_SIZE_LIMIT,
            "timeout_ms": SEARCH_TIMEOUT_MS,
        },
    }
    return planner.plan(PLANNER_SYSTEM_PROMPT, payload)


# -----------------------------------------------------------------------------
# Heuristic planner
# -----------------------------------------------------------------------------


def detect_date_range(nl: str) -> Optional[Dict[str, str]]:
    """Detect simple natural language date ranges."""

    text = nl.lower()
    now = "now"
    if "today" in text:
        return {"gte": "now/d", "lte": now}
    if "yesterday" in text:
        return {"gte": "now-1d/d", "lte": "now-1d/d"}
    if "last 24 hours" in text:
        return {"gte": "now-24h", "lte": now}
    if "last week" in text:
        return {"gte": "now-7d", "lte": now}
    if "last month" in text:
        return {"gte": "now-30d", "lte": now}
    if "this quarter" in text:
        return {"gte": "now-90d", "lte": now}
    return None


def choose_indices_for_question(nl: str) -> List[str]:
    """Pick indices when host does not supply them."""

    client = get_es_client()
    try:
        cat_indices = client.cat.indices(format="json")
    except es_exceptions.ElasticsearchException:
        return ["*"]

    lowered = nl.lower()
    matches: List[str] = []
    for entry in cat_indices:
        name = entry.get("index") or ""
        if not name:
            continue
        if "log" in lowered and "log" in name:
            matches.append(name)
        elif "event" in lowered and "event" in name:
            matches.append(name)
        elif "order" in lowered and ("order" in name or "checkout" in lowered):
            matches.append(name)
    if matches:
        return matches
    return [entry.get("index") for entry in cat_indices if entry.get("index")] or ["*"]


def build_heuristic_query(
    nl: str,
    indices: Optional[List[str]],
) -> Dict[str, Any]:
    """Construct a basic DSL query using simple heuristics."""

    chosen_indices = indices or choose_indices_for_question(nl)
    field_inventory = extract_field_inventory(chosen_indices)
    text_fields = [f for f in field_inventory if f.get("isText")]
    keyword_fields = [f for f in field_inventory if f.get("hasKeyword")]

    bool_query: Dict[str, Any] = {"must": [], "filter": []}

    # Add text search across common fields.
    if text_fields:
        text_field_names = [f["name"] for f in text_fields][:5]
        bool_query["must"].append(
            {
                "multi_match": {
                    "query": nl,
                    "fields": text_field_names,
                }
            }
        )
    else:
        bool_query["must"].append({"match_all": {}})

    # Date filter handling.
    detected_range = detect_date_range(nl)
    if detected_range is None:
        detected_range = {"gte": "now-30d", "lte": "now"}
    bool_query["filter"].append({"range": {DEFAULT_TIME_FIELD: detected_range}})

    dsl: Dict[str, Any] = {
        "query": {"bool": bool_query},
        "size": min(10, SEARCH_SIZE_LIMIT),
    }

    lower_nl = nl.lower()
    assumptions: List[str] = []

    if "count" in lower_nl or "total" in lower_nl:
        dsl["size"] = 0
        assumptions.append("Counting matches, size set to 0.")

    if "top" in lower_nl or "most common" in lower_nl:
        dsl["size"] = 0
        agg_field = None
        if keyword_fields:
            agg_field = keyword_fields[0]["name"]
        elif text_fields:
            agg_field = text_fields[0]["name"]
        if agg_field:
            dsl["aggs"] = {
                "top_values": {
                    "terms": {
                        "field": agg_field,
                        "size": 5,
                    }
                }
            }
            assumptions.append(f"Using {agg_field} for terms aggregation.")

    if "trend" in lower_nl or "per" in lower_nl and "hour" in lower_nl:
        dsl["size"] = 0
        interval = "1h"
        if "day" in lower_nl:
            interval = "1d"
        dsl.setdefault("aggs", {})["trend"] = {
            "date_histogram": {
                "field": DEFAULT_TIME_FIELD,
                "fixed_interval": interval,
            }
        }
        assumptions.append("Added date_histogram for trend analysis.")

    try:
        validated = validate_and_prepare_dsl(dsl)
    except ValueError:
        validated = dsl

    return {
        "indices": chosen_indices,
        "dsl": validated,
        "confidence": 0.5,
        "assumptions": assumptions,
        "alternatives": [
            {"description": "Try match_phrase for exact wording."},
            {"description": "Consider widening the date range."},
        ],
    }


def plan_query_internal(
    nl: str,
    indices: Optional[List[str]],
) -> Dict[str, Any]:
    """Plan a query using LLM or heuristic fallback."""

    chosen_indices = indices or []
    field_inventory = extract_field_inventory(chosen_indices or choose_indices_for_question(nl))

    if get_planner_client():
        llm_result = call_llm_planner(nl, chosen_indices, field_inventory)
        if llm_result and isinstance(llm_result.get("dsl"), dict):
            try:
                llm_result["dsl"] = validate_and_prepare_dsl(llm_result["dsl"])
                return llm_result
            except ValueError:
                pass

    return build_heuristic_query(nl, indices)


# -----------------------------------------------------------------------------
# MCP tools
# -----------------------------------------------------------------------------


@server.tool()
def list_indices(prefix: str = "") -> Dict[str, Any]:
    """List indices filtered by optional prefix."""

    client = get_es_client()
    try:
        response = client.cat.indices(format="json")
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to list indices: {exc}"}

    indices: List[Dict[str, Any]] = []
    for entry in response:
        name = entry.get("index")
        if not name:
            continue
        if prefix and not name.startswith(prefix):
            continue
        docs_count = entry.get("docs.count") or entry.get("docsCount") or "0"
        try:
            docs = int(docs_count)
        except ValueError:
            docs = 0
        store_size = parse_byte_size(entry.get("store.size"))
        indices.append({
            "name": name,
            "docs": docs,
            "store_bytes": store_size,
        })
    return {"indices": indices}


@server.tool()
def get_mapping(index: str) -> Dict[str, Any]:
    """Return flattened field information for an index."""

    cache_key = f"mapping:{index}"
    cached = mappings_cache.get(cache_key)
    if cached is not None:
        return {"fields": cached}

    client = get_es_client()
    try:
        response = client.indices.get_mapping(index=index)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to fetch mapping: {exc}"}

    index_data = response.get(index)
    if not index_data:
        return {"fields": []}

    flattened = flatten_mappings(index_data.get("mappings", {}))
    mappings_cache.set(cache_key, flattened)
    return {"fields": flattened}


@server.tool()
def get_field_caps(indices: List[str]) -> Dict[str, Any]:
    """Return field capabilities for the provided indices."""

    if not indices:
        return {"error": "indices list must not be empty."}

    key = "fieldcaps:" + ",".join(sorted(indices))
    cached = field_caps_cache.get(key)
    if cached is not None:
        return {"caps": cached}

    client = get_es_client()
    try:
        response = client.field_caps(index=indices, fields="*")
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to fetch field caps: {exc}"}

    summary = summarize_field_caps(response)
    field_caps_cache.set(key, summary)
    return {"caps": summary}


@server.tool()
def sample_values(index: str, field: str, size: int = 10) -> Dict[str, Any]:
    """Return sample values for a field using a terms aggregation."""

    client = get_es_client()
    agg_size = max(1, min(size, 20))
    body = {
        "size": 0,
        "aggs": {
            "samples": {
                "terms": {
                    "field": field,
                    "size": agg_size,
                    "execution_hint": "map",
                }
            }
        },
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }
    try:
        response = client.search(index=index, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to sample values: {exc}"}

    buckets = (
        response.get("aggregations", {})
        .get("samples", {})
        .get("buckets", [])
    )
    examples = [bucket.get("key") for bucket in buckets]
    return {"examples": examples}


@server.tool()
def execute_search(
    index: Union[str, List[str]],
    dsl: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute an Elasticsearch search after validating the DSL."""

    try:
        prepared = validate_and_prepare_dsl(dsl)
    except ValueError as exc:
        return {"error": str(exc)}

    client = get_es_client()
    try:
        response = client.search(index=index, body=prepared)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Search failed: {exc}"}

    hits_section = response.get("hits", {})
    raw_hits = hits_section.get("hits", [])
    hits: List[Dict[str, Any]] = []
    for hit in raw_hits:
        hits.append(
            {
                "_id": hit.get("_id"),
                "_index": hit.get("_index"),
                "_score": hit.get("_score"),
                "_source": hit.get("_source"),
            }
        )

    total_value = hits_section.get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0

    return {
        "tookMs": response.get("took", 0),
        "total": total_count,
        "hits": hits,
        "aggs": response.get("aggregations"),
        "timed_out": response.get("timed_out", False),
    }


if ENABLE_PLANNER:
    @server.tool()
    def plan_query(
        nl: str,
        indices: Optional[List[str]] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Plan a query using an LLM (if configured) or heuristics."""

        if not nl:
            return {"error": "nl must be provided."}
        plan = plan_query_internal(nl, indices)
        return plan


# -----------------------------------------------------------------------------
# Server start
# -----------------------------------------------------------------------------


def log_startup() -> None:
    """Log configuration at startup."""

    tools_enabled = ["list_indices", "get_mapping", "get_field_caps", "sample_values", "execute_search"]
    if ENABLE_PLANNER:
        tools_enabled.append("plan_query")
    print(
        "[mcp-elastic] Starting with ES URL="
        f"{ES_URL} api_key={'yes' if ES_API_KEY else 'no'} tools={','.join(tools_enabled)}"
    )
    print("[mcp-elastic] Typical flow: list_indices -> get_mapping -> execute_search")
    if ENABLE_PLANNER:
        print("[mcp-elastic] Planner flow: plan_query -> execute_search")


if __name__ == "__main__":
    log_startup()
    #server.run(transport="streamable-http")
    server.run(transport="stdio")
