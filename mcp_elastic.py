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
import logging

load_dotenv(find_dotenv())


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
# Set default level to DEBUG for development, can be overridden by LOG_LEVEL env var
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)






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

DEFAULT_LOG_FIELDS = {
    "trace_id": "trace_id",
    "span_id": "span_id",
    "message": "log_message",
    "service": "service_id",
    "user": "userid",
    "status": "response_status",
    "response_time": "response_time",
    "path": "path",
    "time": DEFAULT_TIME_FIELD,
}


def discover_log_fields(
    inventory: List[Dict[str, Any]],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Dynamically discover common log fields from inventory."""
    # Define candidate field names for each role
    field_candidates = {
        "service": ["service.name", "service", "app", "service_id"],
        "user": ["user.id", "user.name", "user", "userid", "customer_id"],
        "status": [
            "http.status_code",
            "response.status_code",
            "status",
            "response_status",
            "status_code",
        ],
        "response_time": [
            "event.duration",
            "response_time",
            "duration",
            "latency",
            "responsetime",
        ],
        "path": ["http.request.path", "url.path", "path", "request_path"],
        "message": ["message", "log", "log_message", "error.message"],
        "trace_id": ["trace.id", "trace_id"],
        "span_id": ["span.id", "span_id"],
        "time": ["@timestamp", "timestamp", "event.created"],
    }

    discovered_fields: Dict[str, str] = {}
    inventory_field_names = {f["name"] for f in inventory}

    for role, candidates in field_candidates.items():
        # Find first candidate that exists in the inventory
        for candidate in candidates:
            if candidate in inventory_field_names:
                discovered_fields[role] = candidate
                break

    # Time field is important, ensure it has a default from environment
    if "time" not in discovered_fields:
        discovered_fields["time"] = DEFAULT_TIME_FIELD

    # User overrides take precedence
    if overrides:
        for key, value in overrides.items():
            if key in field_candidates and value:
                discovered_fields[key] = value

    return discovered_fields


def discover_fields_with_llm(inventory: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Use an LLM to discover field mappings based on semantic roles."""

    planner = get_planner_client()
    if not planner:
        return None

    semantic_roles = [
        "service", "user", "status", "response_time", "path", "message", "trace_id", "span_id", "time"
    ]
    
    field_details = "\n".join(
        sorted([f"- {f.get('name')} (type: {f.get('type')})" for f in inventory])
    )
    
    system_prompt = (
        "You are an expert Elasticsearch data analyst. Your task is to map semantic concepts to the most "
        "appropriate fields from a provided list of Elasticsearch index fields.\n\n"
        "The semantic concepts are: "
        f"{', '.join(semantic_roles)}.\n\n"
        "Analyze the field list below and return a single JSON object where the keys are the concepts "
        "and the values are the best matching field names from the list. If no suitable field is found for a concept, "
        "omit it from the JSON object. Your output must be a single, valid JSON object and nothing else.\n\n"
        "Available fields:\n"
        f"{field_details}"
    )

    try:
        # The planner client is expected to handle the LLM interaction.
        discovered_fields = planner.plan(system_prompt, {})
        
        if not discovered_fields or not isinstance(discovered_fields, dict):
            return None

        # Validate that the returned fields are valid and expected.
        inventory_field_names = {f["name"] for f in inventory}
        final_fields = {
            role: field_name
            for role, field_name in discovered_fields.items()
            if role in semantic_roles and isinstance(field_name, str) and field_name in inventory_field_names
        }
            
        return final_fields if final_fields else None

    except Exception:
        # Gracefully fail if LLM call or parsing fails.
        return None


def resolve_log_fields(overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge user-supplied field overrides with defaults."""

    fields = dict(DEFAULT_LOG_FIELDS)
    if overrides:
        for key, value in overrides.items():
            if key in fields and value:
                fields[key] = value
    if not fields.get("time"):
        fields["time"] = DEFAULT_TIME_FIELD
    return fields


def get_by_dotted_path(source: Dict[str, Any], path: Optional[str]) -> Any:
    """Return a nested field from _source using dotted notation."""

    if not path:
        return None
    parts = path.split(".")
    current: Any = source
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def find_field_info(field_name: Optional[str], inventory: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Locate metadata about a field from the flattened mappings."""

    if not field_name:
        return None
    return next((f for f in inventory if f.get("name") == field_name), None)


def ensure_terms_field(field_name: Optional[str], inventory: List[Dict[str, Any]]) -> Optional[str]:
    """Return a field suitable for terms aggregations, falling back to .keyword when available."""

    if not field_name:
        return None
    if field_name.endswith(".keyword"):
        return field_name
    info = find_field_info(field_name, inventory)
    if info and info.get("hasKeyword") and info.get("type") != "keyword":
        return f"{field_name}.keyword"
    return field_name


def strip_keyword_suffix(field_name: Optional[str]) -> Optional[str]:
    """Trim trailing .keyword so _source lookups work as expected."""

    if not field_name:
        return None
    if field_name.endswith(".keyword"):
        return field_name[: -len(".keyword")]
    return field_name
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


def normalize_indices_param(index: Union[str, List[str]]) -> List[str]:
    """Normalize index parameter into a non-empty list when possible."""

    if isinstance(index, str):
        indices = [index]
    else:
        indices = list(index)
    return [name for name in indices if name]


def parse_terms_buckets(buckets: List[Dict[str, Any]], value_key: str = "key") -> List[Dict[str, Any]]:
    """Convert Elasticsearch terms buckets into compact summaries."""

    parsed: List[Dict[str, Any]] = []
    for bucket in buckets:
        parsed.append(
            {
                "value": bucket.get(value_key),
                "count": bucket.get("doc_count", 0),
            }
        )
    return parsed


def build_log_sample(source: Dict[str, Any], fields: Dict[str, str]) -> Dict[str, Any]:
    """Extract representative log fields from a hit source."""

    sample: Dict[str, Any] = {}
    for label, field_key in [
        ("trace_id", "trace_id"),
        ("span_id", "span_id"),
        ("service", "service"),
        ("user", "user"),
        ("status", "status"),
        ("response_time", "response_time"),
        ("path", "path"),
    ]:
        field_name = strip_keyword_suffix(fields.get(field_key))
        value = get_by_dotted_path(source, field_name)
        if value is not None:
            sample[label] = value

    message_field = strip_keyword_suffix(fields.get("message"))
    message_value = get_by_dotted_path(source, message_field)
    if isinstance(message_value, str):
        sample["message"] = message_value[:500]
    elif message_value is not None:
        sample["message"] = message_value
    return sample


def field_is_numeric(field_name: Optional[str], inventory: List[Dict[str, Any]]) -> bool:
    """Return True when the field is mapped as numeric."""

    info = find_field_info(field_name, inventory)
    return bool(info and info.get("isNumeric"))
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


def build_time_filters(lookback: Optional[str], time_field: str) -> List[Dict[str, Any]]:
    """Create range filters for the requested lookback window."""

    if not lookback:
        return []
    return [{"range": {time_field: {"gte": lookback, "lte": "now"}}}]


def extract_field_inventory_for_indices(index: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Return mapping inventory for the first index in the list."""

    indices = normalize_indices_param(index)
    if not indices:
        return []
    return extract_field_inventory(indices)


@server.tool()
def summarize_logs(
    index: Union[str, List[str]],
    lookback: str = "now-24h",
    field_overrides: Optional[Dict[str, str]] = None,
    include_samples: bool = True,
    sample_size: int = 3,
    sample_strategy: str = "recent",
    max_message_length: int = 500,
) -> Dict[str, Any]:
    """Get high-level log summary (error rates, top services) with optional sampling.

    Args:
        index: Index name/pattern (e.g., "logs-*").
        lookback: Time window (e.g., "now-24h").
        field_overrides: Optional field mapping overrides.
        include_samples: Include log samples (default: True).
        sample_size: Number of samples (0-10).
        sample_strategy: "recent", "random", or "none".
        max_message_length: Truncate messages to this length.
    """

    logger.debug(f"summarize_logs called with index={index}, lookback={lookback}, include_samples={include_samples}")

    indices = normalize_indices_param(index)
    if not indices:
        logger.warning("No indices provided, returning error.")
        return {"error": "index must be provided."}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    logger.debug(f"Inventory contains {len(inventory)} fields.")

    fields = None
    # Try LLM-powered discovery first
    if ENABLE_PLANNER:
        logger.debug("ENABLE_PLANNER is True. Attempting LLM-powered field discovery.")
        fields = discover_fields_with_llm(inventory)
        if fields:
            logger.debug(f"LLM-powered discovery successful. Fields: {fields}")
        else:
            logger.debug("LLM-powered discovery failed or returned no fields.")

    # Fallback to heuristic-based discovery if LLM fails or is disabled
    if not fields:
        logger.debug("Falling back to heuristic field discovery.")
        fields = discover_log_fields(inventory)
        logger.debug(f"Heuristic discovery fields: {fields}")

    # Apply user overrides as the final step
    if field_overrides:
        logger.debug(f"Applying field overrides: {field_overrides}")
        fields.update(field_overrides)
    
    logger.debug(f"Final resolved fields for aggregations: {fields}")

    service_field = ensure_terms_field(fields.get("service"), inventory)
    user_field = ensure_terms_field(fields.get("user"), inventory)
    status_field = fields.get("status")
    status_terms_field = ensure_terms_field(status_field, inventory)
    path_field = fields.get("path")
    path_terms_field = ensure_terms_field(path_field, inventory)
    response_time_field = fields.get("response_time")
    response_time_numeric = field_is_numeric(response_time_field, inventory)
    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD

    logger.debug(f"Aggregation fields - service_field={service_field}, user_field={user_field}, status_terms_field={status_terms_field}, path_terms_field={path_terms_field}, response_time_field={response_time_field}, time_field={time_field}")

    query_filters = build_time_filters(lookback, time_field)
    bool_query: Dict[str, Any] = {"must": [], "filter": query_filters} if query_filters else {"must": [], "filter": []}
    if not query_filters:
        bool_query["must"].append({"match_all": {}})
    logger.debug(f"Query filters: {query_filters}")

    aggs: Dict[str, Any] = {}
    if status_terms_field:
        # Check if status field is numeric to avoid "UNKNOWN" string error
        status_is_numeric = field_is_numeric(status_field, inventory)
        terms_agg = {
            "field": status_terms_field,
            "size": 6,
        }
        # Only add missing value if field is not numeric (string missing values fail on numeric fields)
        if not status_is_numeric:
            terms_agg["missing"] = "UNKNOWN"
        else:
            # For numeric fields, use -1 to represent unknown/missing status codes
            terms_agg["missing"] = -1

        aggs["status_counts"] = {"terms": terms_agg}
        logger.debug(f"Added status_counts aggregation for field: {status_terms_field} (numeric={status_is_numeric})")

    if service_field:
        aggs["top_services"] = {
            "terms": {
                "field": service_field,
                "size": 5,
                "missing": "unknown_service",
            }
        }
        logger.debug(f"Added top_services aggregation for field: {service_field}")

    if user_field:
        aggs["top_users"] = {
            "terms": {
                "field": user_field,
                "size": 5,
                "missing": "anonymous",
            }
        }
        logger.debug(f"Added top_users aggregation for field: {user_field}")

    if response_time_numeric and response_time_field:
        aggs["response_time_stats"] = {"stats": {"field": response_time_field}}
        aggs["response_time_percentiles"] = {
            "percentiles": {
                "field": response_time_field,
                "percents": [50, 90, 95, 99],
            }
        }
        logger.debug(f"Added response_time aggregations for field: {response_time_field}")

    if path_terms_field and response_time_field and response_time_numeric:
        aggs["slow_paths"] = {
            "terms": {
                "field": path_terms_field,
                "size": 5,
                "order": {"avg_latency": "desc"},
                "missing": "unknown_path",
            },
            "aggs": {
                "avg_latency": {"avg": {"field": response_time_field}},
            },
        }
        logger.debug(f"Added slow_paths aggregation for field: {path_terms_field}")

    if status_field and field_is_numeric(status_field, inventory):
        aggs["status_buckets"] = {
            "filters": {
                "filters": {
                    "errors": {"range": {status_field: {"gte": 500}}},
                    "warnings": {"range": {status_field: {"gte": 400, "lt": 500}}},
                    "success": {"range": {status_field: {"gte": 200, "lt": 400}}},
                }
            }
        }
        logger.debug(f"Added status_buckets aggregation for field: {status_field}")

    # Determine actual sampling behavior
    actual_sample_size = 0
    if sample_strategy == "none" or sample_size <= 0:
        actual_sample_size = 0
        include_samples = False
    elif include_samples:
        actual_sample_size = max(0, min(sample_size, 10))  # Clamp to 0-10

    body: Dict[str, Any] = {
        "size": actual_sample_size,
        "query": {"bool": bool_query},
        "aggs": aggs,
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
        "track_total_hits": True,
    }

    if actual_sample_size > 0:
        if sample_strategy == "random":
            # Use random_score for unbiased sampling
            body["query"] = {
                "function_score": {
                    "query": body["query"],
                    "random_score": {},
                    "boost_mode": "replace",
                }
            }
            logger.debug("Using random sampling strategy")
        else:  # "recent" or default
            body["sort"] = [{time_field: {"order": "desc"}}]
            logger.debug(f"Using recent sampling strategy, sorted by {time_field}")

    logger.debug(f"Elasticsearch search body: {json.dumps(body, indent=2)}")

    try:
        response = client.search(index=indices, body=body)
        logger.debug(f"Elasticsearch search successful. Took: {response.get('took')}ms")
        # Print a truncated version of the response to avoid excessive output
        logger.debug(f"Elasticsearch response (truncated, first 1000 chars): {json.dumps(response, indent=2)[:1000]}...")
    except es_exceptions.ElasticsearchException as exc:
        logger.error(f"Elasticsearch search failed: {exc}", exc_info=True)
        return {"error": f"Failed to summarize logs: {exc}"}

    hits_section = response.get("hits", {}) or {}
    total_value = hits_section.get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0
    logger.debug(f"Total count from hits_section: {total_count}")

    aggregations = response.get("aggregations") or {}
    logger.debug(f"Aggregations section received: {list(aggregations.keys()) if aggregations else 'None'}")

    status_counts = parse_terms_buckets(
        aggregations.get("status_counts", {}).get("buckets", [])
    ) if "status_counts" in aggregations else []
    logger.debug(f"Status counts: {status_counts}")

    status_buckets = aggregations.get("status_buckets", {}).get("buckets", {})
    errors = status_buckets.get("errors", {}).get("doc_count", 0)
    warnings = status_buckets.get("warnings", {}).get("doc_count", 0)
    success = status_buckets.get("success", {}).get("doc_count", 0)
    error_rate = (errors / total_count) if (total_count and status_buckets) else 0.0
    logger.debug(f"Status buckets - errors={errors}, warnings={warnings}, success={success}, error_rate={error_rate}")

    top_services = parse_terms_buckets(
        aggregations.get("top_services", {}).get("buckets", [])
    ) if "top_services" in aggregations else []
    logger.debug(f"Top services: {top_services}")

    top_users = parse_terms_buckets(
        aggregations.get("top_users", {}).get("buckets", [])
    ) if "top_users" in aggregations else []
    logger.debug(f"Top users: {top_users}")

    slow_paths: List[Dict[str, Any]] = []
    if "slow_paths" in aggregations:
        for bucket in aggregations["slow_paths"].get("buckets", []):
            slow_paths.append(
                {
                    "path": bucket.get("key"),
                    "count": bucket.get("doc_count", 0),
                    "avg_response_time": bucket.get("avg_latency", {}).get("value"),
                }
            )
    logger.debug(f"Slow paths: {slow_paths}")

    samples: List[Dict[str, Any]] = []
    if include_samples and actual_sample_size > 0:
        for hit in hits_section.get("hits", []):
            source = hit.get("_source") or {}
            if isinstance(source, dict):
                sample = build_log_sample(source, fields)
                # Apply message truncation
                if "message" in sample and isinstance(sample["message"], str):
                    if len(sample["message"]) > max_message_length:
                        sample["message"] = sample["message"][:max_message_length] + "..."
                samples.append(sample)
    logger.debug(f"Number of samples generated: {len(samples)}")

    status_summary: Dict[str, Any] = {"distribution": status_counts}
    if status_buckets:
        status_summary["groups"] = {
            "errors": errors,
            "warnings": warnings,
            "success": success,
            "error_rate": round(error_rate, 4),
        }

    result: Dict[str, Any] = {
        "total": total_count,
        "time_window": {"gte": lookback, "lte": "now"} if lookback else None,
        "status": status_summary,
        "services": top_services,
        "users": top_users,
        "slow_paths": slow_paths,
    }

    if response_time_numeric and response_time_field:
        stats = aggregations.get("response_time_stats", {})
        percentiles = aggregations.get("response_time_percentiles", {}).get("values", {})
        result["response_time"] = {
            "avg": stats.get("avg"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "p50": percentiles.get("50.0"),
            "p90": percentiles.get("90.0"),
            "p95": percentiles.get("95.0"),
            "p99": percentiles.get("99.0"),
        }
        logger.debug(f"Response time stats added: {result.get('response_time')}")

    if samples:
        result["samples"] = samples
        result["sampling_metadata"] = {
            "strategy": sample_strategy,
            "requested_size": sample_size,
            "returned_size": len(samples),
            "max_message_length": max_message_length,
        }
        logger.debug("Samples added to result.")
    elif sample_strategy != "none" and sample_size > 0:
        # Indicate that sampling was requested but no samples were found
        result["sampling_metadata"] = {
            "strategy": sample_strategy,
            "requested_size": sample_size,
            "returned_size": 0,
            "note": "No samples matched the query criteria",
        }

    logger.debug(f"Final result keys: {list(result.keys())}")
    return result


@server.tool()
def log_trend(
    index: Union[str, List[str]],
    lookback: str = "now-24h",
    interval: str = "1h",
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Get log volume and error trends over time buckets.

    Args:
        index: Index name/pattern.
        lookback: Total time window (e.g., "now-24h").
        interval: Bucket size (e.g., "1h").
        field_overrides: Optional field mapping overrides.
    """

    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided."}

    client = get_es_client()
    fields = resolve_log_fields(field_overrides)
    inventory = extract_field_inventory_for_indices(indices)

    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD
    status_field = ensure_terms_field(fields.get("status"), inventory)
    response_time_field = fields.get("response_time")
    response_time_numeric = field_is_numeric(response_time_field, inventory)

    aggs: Dict[str, Any] = {
        "by_interval": {
            "date_histogram": {
                "field": time_field,
                "fixed_interval": interval,
                "format": "strict_date_optional_time",
                "min_doc_count": 0,
            },
            "aggs": {},
        }
    }

    if status_field:
        aggs["by_interval"]["aggs"]["status_counts"] = {
            "terms": {
                "field": status_field,
                "size": 5,
                "missing": "UNKNOWN",
            }
        }

    if response_time_field and response_time_numeric:
        aggs["by_interval"]["aggs"]["avg_response_time"] = {
            "avg": {"field": response_time_field}
        }

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": build_time_filters(lookback, time_field),
            }
        },
        "aggs": aggs,
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to fetch log trend: {exc}"}

    buckets = (
        response.get("aggregations", {})
        .get("by_interval", {})
        .get("buckets", [])
    )

    trend: List[Dict[str, Any]] = []
    for bucket in buckets:
        entry: Dict[str, Any] = {
            "interval_start": bucket.get("key_as_string"),
            "count": bucket.get("doc_count", 0),
        }
        status_aggs = bucket.get("status_counts", {})
        if status_aggs:
            entry["status_distribution"] = parse_terms_buckets(status_aggs.get("buckets", []))
        avg_latency = bucket.get("avg_response_time", {}).get("value")
        if avg_latency is not None:
            entry["avg_response_time"] = avg_latency
        trend.append(entry)

    return {
        "time_window": {"gte": lookback, "lte": "now"} if lookback else None,
        "interval": interval,
        "buckets": trend,
    }


@server.tool()
def sample_trace(
    index: Union[str, List[str]],
    trace_id: str,
    size: int = 10,
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Get events for a specific trace ID to visualize request journey.

    Args:
        index: Index name/pattern.
        trace_id: Unique trace identifier.
        size: Max events (default: 10).
        field_overrides: Optional field mapping overrides.
    """

    if not trace_id:
        return {"error": "trace_id must be provided."}

    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided."}

    client = get_es_client()
    fields = resolve_log_fields(field_overrides)
    inventory = extract_field_inventory_for_indices(indices)

    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD
    trace_source_field = fields.get("trace_id")
    if not trace_source_field:
        return {"error": "trace_id field is not configured."}

    trace_field = ensure_terms_field(trace_source_field, inventory) or trace_source_field
    span_field = ensure_terms_field(fields.get("span_id"), inventory)
    status_field = ensure_terms_field(fields.get("status"), inventory)

    fetch_size = max(1, min(size, 50))
    query = {
        "bool": {
            "filter": [
                {"term": {trace_field: {"value": trace_id}}},
            ]
        }
    }

    body: Dict[str, Any] = {
        "size": fetch_size,
        "query": query,
        "sort": [{time_field: {"order": "asc"}}],
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
        "aggs": {},
        "track_total_hits": True,
    }

    if span_field:
        body["aggs"]["span_counts"] = {
            "terms": {"field": span_field, "size": fetch_size}
        }

    if status_field:
        # Check if status field is numeric to avoid "For input string: 'UNKNOWN'" error
        is_numeric_status = False
        status_info = find_field_info(status_field, inventory)
        if status_info and status_info.get("isNumeric"):
            is_numeric_status = True
        
        missing_val = -1 if is_numeric_status else "UNKNOWN"
        
        body["aggs"]["status_counts"] = {
            "terms": {"field": status_field, "size": 5, "missing": missing_val}
        }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to sample trace: {exc}"}

    hits_section = response.get("hits", {}) or {}
    total_value = hits_section.get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0

    samples: List[Dict[str, Any]] = []
    for hit in hits_section.get("hits", []):
        source = hit.get("_source") or {}
        if isinstance(source, dict):
            samples.append(build_log_sample(source, fields))

    aggs = response.get("aggregations", {}) or {}
    span_counts = parse_terms_buckets(aggs.get("span_counts", {}).get("buckets", []))
    status_counts = parse_terms_buckets(aggs.get("status_counts", {}).get("buckets", []))

    return {
        "trace_id": trace_id,
        "total_events": total_count,
        "span_counts": span_counts,
        "status_distribution": status_counts,
        "samples": samples,
    }


@server.tool()
def list_indices(prefix: str = "") -> Dict[str, Any]:
    """List indices with doc counts and size.

    Args:
        prefix: Filter by index name prefix.
    """

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
    """Get flattened field mapping for an index.

    Args:
        index: Index name.
    """

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
    """Check if fields are searchable/aggregatable.

    Args:
        indices: List of index names.
    """

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
    """Get unique value samples for a field.

    Args:
        index: Index name.
        field: Field name (use .keyword for text).
        size: Max samples (default: 10).
    """

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
    """Execute raw Elasticsearch DSL query.

    Args:
        index: Index name/pattern.
        dsl: Query DSL dict (query, aggs, size, etc).
    """

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


@server.tool()
def search_with_projection(
    index: Union[str, List[str]],
    dsl: Dict[str, Any],
    fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search with field projection to reduce size.

    Args:
        index: Index name/pattern.
        dsl: Query DSL.
        fields: Fields to include.
        exclude_fields: Fields to exclude.
    """

    try:
        prepared = validate_and_prepare_dsl(dsl)
    except ValueError as exc:
        return {"error": str(exc)}

    # Add field projection to DSL
    if fields:
        prepared["_source"] = fields
    elif exclude_fields:
        prepared["_source"] = {"excludes": exclude_fields}

    client = get_es_client()
    try:
        response = client.search(index=index, body=prepared)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Search failed: {exc}"}

    hits_section = response.get("hits", {})
    raw_hits = hits_section.get("hits", [])
    hits: List[Dict[str, Any]] = []

    for hit in raw_hits:
        hits.append({
            "_id": hit.get("_id"),
            "_index": hit.get("_index"),
            "_score": hit.get("_score"),
            "_source": hit.get("_source"),  # Now contains only requested fields
        })

    total_value = hits_section.get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0

    # Add metadata about response size reduction
    metadata = {
        "projected_fields": fields,
        "excluded_fields": exclude_fields,
        "documents_returned": len(hits),
    }

    return {
        "tookMs": response.get("took", 0),
        "total": total_count,
        "hits": hits,
        "aggs": response.get("aggregations"),
        "timed_out": response.get("timed_out", False),
        "metadata": metadata,
    }


@server.tool()
def count_and_aggregate(
    index: Union[str, List[str]],
    query: Optional[Dict[str, Any]] = None,
    aggregations: Optional[Dict[str, Any]] = None,
    time_range: Optional[Dict[str, str]] = None,
    time_field: str = "@timestamp",
) -> Dict[str, Any]:
    """Get stats/counts without documents (fast, low overhead).

    Args:
        index: Index name/pattern.
        query: Optional query clause.
        aggregations: Aggregation definitions.
        time_range: Optional time filter (gte/lte).
        time_field: Time field name.
    """

    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    # Build query
    bool_query: Dict[str, Any] = {"must": [], "filter": []}

    if query:
        bool_query["must"].append(query)
    else:
        bool_query["must"].append({"match_all": {}})

    if time_range:
        bool_query["filter"].append({"range": {time_field: time_range}})

    # Build request body - size=0 means NO documents returned
    body: Dict[str, Any] = {
        "size": 0,  # CRITICAL: No documents
        "query": {"bool": bool_query},
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
        "track_total_hits": True,
    }

    if aggregations:
        body["aggs"] = aggregations

    client = get_es_client()
    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Aggregation failed: {exc}"}

    total_value = response.get("hits", {}).get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0

    return {
        "tookMs": response.get("took", 0),
        "total_count": total_count,
        "aggregations": response.get("aggregations", {}),
        "timed_out": response.get("timed_out", False),
        "metadata": {
            "documents_returned": 0,
            "aggregations_only": True,
            "query_applied": query is not None,
            "time_filter_applied": time_range is not None,
        }
    }


@server.tool()
def search_paginated(
    index: Union[str, List[str]],
    dsl: Dict[str, Any],
    page_size: int = 10,
    search_after: Optional[List[Any]] = None,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Paginated search using search_after for large sets.

    Args:
        index: Index name/pattern.
        dsl: Query DSL (must include sort).
        page_size: Results per page (max 100).
        search_after: Token from previous page.
        fields: Optional field projection.
    """

    try:
        prepared = validate_and_prepare_dsl(dsl)
    except ValueError as exc:
        return {"error": str(exc)}

    # Validate sort clause exists
    if "sort" not in prepared:
        return {
            "error": (
                "sort clause required for pagination. "
                "Add a sort clause to your DSL to enable pagination. "
                "Recommended: 'sort': [{'@timestamp': 'desc'}, {'_id': 'desc'}]"
            )
        }

    # Clamp page size
    actual_page_size = max(1, min(page_size, 100))
    prepared["size"] = actual_page_size

    # Add search_after for pagination (skip on first page)
    if search_after:
        prepared["search_after"] = search_after

    # Add field projection if specified
    if fields:
        prepared["_source"] = fields

    client = get_es_client()
    try:
        response = client.search(index=index, body=prepared)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Search failed: {exc}"}

    hits_section = response.get("hits", {})
    raw_hits = hits_section.get("hits", [])
    hits: List[Dict[str, Any]] = []
    next_page_token = None

    for hit in raw_hits:
        hits.append({
            "_id": hit.get("_id"),
            "_index": hit.get("_index"),
            "_score": hit.get("_score"),
            "_source": hit.get("_source"),
        })

    # Extract search_after value from last hit for next page
    if raw_hits:
        last_hit = raw_hits[-1]
        next_page_token = last_hit.get("sort")

    total_value = hits_section.get("total")
    if isinstance(total_value, dict):
        total_count = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_count = total_value
    else:
        total_count = 0

    # Determine if there's a next page
    has_next_page = len(hits) == actual_page_size and next_page_token is not None

    return {
        "tookMs": response.get("took", 0),
        "total": total_count,
        "hits": hits,
        "page_size": actual_page_size,
        "documents_in_page": len(hits),
        "has_next_page": has_next_page,
        "next_page_token": next_page_token,
        "timed_out": response.get("timed_out", False),
        "metadata": {
            "projected_fields": fields,
            "pagination_enabled": True,
            "current_page_size": len(hits),
        }
    }


@server.tool()
def estimate_response_size(
    index: Union[str, List[str]],
    dsl: Dict[str, Any],
    include_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Estimate response size to prevent overload.

    Args:
        index: Index name/pattern.
        dsl: Query DSL.
        include_fields: Optional field projection.
    """

    try:
        prepared = validate_and_prepare_dsl(dsl)
    except ValueError as exc:
        return {"error": str(exc)}

    client = get_es_client()

    # First, get a count of matching documents
    count_body = {
        "query": prepared.get("query", {"match_all": {}}),
        "size": 0,
        "track_total_hits": True,
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        count_response = client.search(index=index, body=count_body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to estimate size: {exc}"}

    total_value = count_response.get("hits", {}).get("total")
    if isinstance(total_value, dict):
        total_matches = total_value.get("value", 0)
    elif isinstance(total_value, int):
        total_matches = total_value
    else:
        total_matches = 0

    if total_matches == 0:
        return {
            "total_matches": 0,
            "estimated_response_kb": 0,
            "safe_to_execute": True,
            "recommendations": ["No documents match the query."],
        }

    # Sample a few documents to estimate average size
    sample_size = min(5, total_matches)
    sample_body = {
        "query": prepared.get("query", {"match_all": {}}),
        "size": sample_size,
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    if include_fields:
        sample_body["_source"] = include_fields

    try:
        sample_response = client.search(index=index, body=sample_body)
    except es_exceptions.ElasticsearchException:
        avg_doc_size_kb = 10.0  # Fallback estimate
    else:
        hits = sample_response.get("hits", {}).get("hits", [])
        if hits:
            total_sample_size = sum(len(json.dumps(hit.get("_source", {}))) for hit in hits)
            avg_doc_size_kb = (total_sample_size / len(hits)) / 1024
        else:
            avg_doc_size_kb = 10.0

    # Calculate estimates
    requested_size = prepared.get("size", 10)
    docs_to_return = min(requested_size, total_matches)
    estimated_response_kb = docs_to_return * avg_doc_size_kb

    # Determine safety
    safe_threshold_kb = 1000  # 1MB threshold
    safe_to_execute = estimated_response_kb < safe_threshold_kb

    # Generate recommendations
    recommendations = []

    if not safe_to_execute:
        recommendations.append(
            f"⚠️ Response may be {estimated_response_kb:.1f}KB ({estimated_response_kb/1024:.2f}MB). "
            "This could overwhelm the MCP host."
        )

        if not include_fields:
            projected_size = estimated_response_kb * 0.1  # Assume 90% reduction
            recommendations.append(
                f"✅ Use search_with_projection to limit fields. "
                f"Estimated size with projection: ~{projected_size:.1f}KB (90% reduction)"
            )

        if requested_size > 50:
            recommendations.append(
                f"✅ Use search_paginated to fetch {requested_size} documents in chunks of 50. "
                f"Each page would be ~{(50 * avg_doc_size_kb):.1f}KB."
            )

        recommendations.append(
            "✅ Use count_and_aggregate if you only need statistics (response: <10KB)."
        )

    if total_matches > requested_size:
        recommendations.append(
            f"ℹ️ Query matches {total_matches} documents but only {requested_size} will be returned. "
            "Use search_paginated for full results or increase 'size' in DSL."
        )

    if not recommendations:
        recommendations.append("✅ Response size is within safe limits. Proceed with query.")

    # Calculate potential savings with field projection
    if not include_fields:
        projected_with_fields = estimated_response_kb * 0.1
        size_savings = {
            "current_estimate_kb": round(estimated_response_kb, 2),
            "with_field_projection_kb": round(projected_with_fields, 2),
            "savings_kb": round(estimated_response_kb - projected_with_fields, 2),
            "savings_percent": 90,
        }
    else:
        size_savings = None

    return {
        "total_matches": total_matches,
        "requested_size": requested_size,
        "docs_to_return": docs_to_return,
        "avg_doc_size_kb": round(avg_doc_size_kb, 2),
        "estimated_response_kb": round(estimated_response_kb, 2),
        "estimated_response_mb": round(estimated_response_kb / 1024, 2),
        "safe_to_execute": safe_to_execute,
        "safe_threshold_kb": safe_threshold_kb,
        "recommendations": recommendations,
        "with_field_projection": include_fields is not None,
        "size_savings": size_savings,
    }


@server.tool()
def sample_logs_stratified(
    index: Union[str, List[str]],
    lookback: str = "now-24h",
    strata_field: str = "response_status",
    samples_per_stratum: int = 2,
    time_field: str = "@timestamp",
    field_overrides: Optional[Dict[str, str]] = None,
    max_strata: int = 10,
) -> Dict[str, Any]:
    """Stratified sampling (e.g. 2 logs per status code).

    Args:
        index: Index name/pattern.
        lookback: Time window.
        strata_field: Field to stratify by (e.g. status).
        samples_per_stratum: Samples per group.
        time_field: Time field.
        field_overrides: Optional overrides.
        max_strata: Max groups.
    """

    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    fields = resolve_log_fields(field_overrides)

    # Resolve the strata field to its aggregatable version
    strata_agg_field = ensure_terms_field(strata_field, inventory) or strata_field

    # Check if the strata field is numeric to handle missing values correctly
    strata_is_numeric = field_is_numeric(strata_field, inventory)
    strata_terms_config = {
        "field": strata_agg_field,
        "size": max_strata * 2,  # Get more to show top strata
        "order": {"_count": "desc"}
    }
    # Add appropriate missing value based on field type
    if strata_is_numeric:
        strata_terms_config["missing"] = -1  # Use -1 for numeric fields
    else:
        strata_terms_config["missing"] = "UNKNOWN"  # Use string for text fields

    # First, get the distribution of the strata field
    agg_body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": build_time_filters(lookback, time_field)
            }
        },
        "aggs": {
            "strata": {
                "terms": strata_terms_config
            }
        },
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        agg_response = client.search(index=indices, body=agg_body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to determine strata: {exc}"}

    strata_buckets = agg_response.get("aggregations", {}).get("strata", {}).get("buckets", [])

    if not strata_buckets:
        return {
            "error": f"No data found for strata field '{strata_field}' in time window",
            "strata_field": strata_field,
            "time_window": {"gte": lookback, "lte": "now"},
        }

    # Now sample from each stratum (limit to max_strata)
    samples_by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    strata_metadata: List[Dict[str, Any]] = []
    total_sampled = 0

    for bucket in strata_buckets[:max_strata]:
        stratum_value = bucket.get("key")
        stratum_count = bucket.get("doc_count", 0)

        if stratum_count == 0:
            continue

        # Query for samples from this stratum
        sample_body = {
            "size": samples_per_stratum,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {strata_agg_field: stratum_value}},
                        *build_time_filters(lookback, time_field)
                    ]
                }
            },
            "sort": [{time_field: {"order": "desc"}}],  # Most recent within stratum
            "timeout": f"{SEARCH_TIMEOUT_MS}ms",
        }

        try:
            sample_response = client.search(index=indices, body=sample_body)
        except es_exceptions.ElasticsearchException:
            continue

        hits = sample_response.get("hits", {}).get("hits", [])
        samples = []
        for hit in hits:
            source = hit.get("_source") or {}
            if isinstance(source, dict):
                samples.append(build_log_sample(source, fields))

        if samples:
            samples_by_stratum[str(stratum_value)] = samples
            total_sampled += len(samples)
            strata_metadata.append({
                "value": str(stratum_value),
                "total_count": stratum_count,
                "samples_retrieved": len(samples),
            })

    # Calculate summary statistics
    total_docs_in_strata = sum(s["total_count"] for s in strata_metadata)

    return {
        "strata_field": strata_field,
        "samples_per_stratum": samples_per_stratum,
        "total_strata_found": len(strata_buckets),
        "strata_sampled": len(samples_by_stratum),
        "total_samples": total_sampled,
        "total_documents_in_strata": total_docs_in_strata,
        "samples": samples_by_stratum,
        "strata_metadata": strata_metadata,
        "time_window": {"gte": lookback, "lte": "now"},
        "sampling_notes": [
            f"Sampled from top {len(samples_by_stratum)} strata by document count",
            f"Each stratum provides up to {samples_per_stratum} samples",
            "Samples within each stratum are the most recent",
        ]
    }


@server.tool()
def get_cluster_health() -> Dict[str, Any]:
    """Check cluster health (status, nodes, shards)."""
    client = get_es_client()
    try:
        health = client.cluster.health()
        return {
            "status": health.get("status"),
            "cluster_name": health.get("cluster_name"),
            "number_of_nodes": health.get("number_of_nodes"),
            "active_shards": health.get("active_shards"),
            "unassigned_shards": health.get("unassigned_shards"),
        }
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to get cluster health: {exc}"}


@server.tool()
def find_traces_by_user(
    index: Union[str, List[str]],
    user_id: str,
    lookback: str = "now-24h",
    size: int = 10,
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Find traces for a user.

    Args:
        index: Index name/pattern.
        user_id: User ID.
        lookback: Time window.
        size: Max traces.
        field_overrides: Optional overrides.
    """
    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    fields = resolve_log_fields(field_overrides)

    user_field = ensure_terms_field(fields.get("user"), inventory)
    trace_field = ensure_terms_field(fields.get("trace_id"), inventory)
    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD

    if not user_field or not trace_field:
        return {"error": "Could not identify user or trace_id fields. Please provide field_overrides."}

    # Aggregation to find unique traces for the user
    body = {
        "size": 0,
        "query": {
            "bool": {
                "must": [{"term": {user_field: user_id}}],
                "filter": build_time_filters(lookback, time_field),
            }
        },
        "aggs": {
            "traces": {
                "terms": {"field": trace_field, "size": size},
                "aggs": {
                    "latest_time": {"max": {"field": time_field}},
                    "services": {
                        "terms": {
                            "field": ensure_terms_field(fields.get("service"), inventory) or "service.keyword",
                            "size": 5
                        }
                    }
                }
            }
        },
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to find traces: {exc}"}

    traces = []
    for bucket in response.get("aggregations", {}).get("traces", {}).get("buckets", []):
        traces.append({
            "trace_id": bucket.get("key"),
            "count": bucket.get("doc_count"),
            "last_seen": bucket.get("latest_time", {}).get("value_as_string"),
            "services": [b.get("key") for b in bucket.get("services", {}).get("buckets", [])]
        })

    return {
        "user_id": user_id,
        "found_traces": len(traces),
        "traces": traces,
        "time_window": {"gte": lookback, "lte": "now"},
    }


@server.tool()
def analyze_error_patterns(
    index: Union[str, List[str]],
    lookback: str = "now-24h",
    size: int = 10,
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Group errors by message to find patterns.

    Args:
        index: Index name/pattern.
        lookback: Time window.
        size: Number of patterns.
        field_overrides: Optional overrides.
    """
    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    fields = resolve_log_fields(field_overrides)

    status_field = fields.get("status")
    message_field = ensure_terms_field(fields.get("message"), inventory)
    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD
    
    # If message is text, we might need a keyword subfield for aggregation
    # If auto-discovery failed to find a keyword version, we might resort to fielddata (risky)
    # or just warn. Here we assume ensure_terms_field did its best.
    
    if not message_field:
         return {"error": "Could not identify a suitable message field for aggregation."}

    status_is_numeric = field_is_numeric(status_field, inventory)
    
    error_filter = None
    if status_is_numeric:
        error_filter = {"range": {status_field: {"gte": 400}}}
    else:
        # Heuristic for string status: matches 4xx or 5xx
        error_filter = {"regexp": {status_field: "[45][0-9]{2}"}}

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    error_filter,
                    *build_time_filters(lookback, time_field)
                ]
            }
        },
        "aggs": {
            "top_errors": {
                "terms": {"field": message_field, "size": size},
                "aggs": {
                    "sample_services": {
                        "terms": {
                            "field": ensure_terms_field(fields.get("service"), inventory) or "service.keyword",
                            "size": 3
                        }
                    }
                }
            }
        },
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to analyze errors: {exc}"}

    patterns = []
    for bucket in response.get("aggregations", {}).get("top_errors", {}).get("buckets", []):
        patterns.append({
            "error_message": bucket.get("key"),
            "count": bucket.get("doc_count"),
            "affected_services": [b.get("key") for b in bucket.get("sample_services", {}).get("buckets", [])]
        })

    return {
        "time_window": {"gte": lookback, "lte": "now"},
        "patterns": patterns,
    }


@server.tool()
def get_metric_statistics(
    index: Union[str, List[str]],
    field: str,
    metric: str = "avg",
    lookback: str = "now-1h",
    interval: Optional[str] = None,
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Calculate numeric metrics (avg, max, min, sum).

    Args:
        index: Index name/pattern.
        field: Numeric field.
        metric: Metric type.
        lookback: Time window.
        interval: Optional bucket size for trend.
        field_overrides: Optional overrides.
    """
    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    fields = resolve_log_fields(field_overrides)
    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD

    # Ensure field exists and get its correct name (though for metrics we usually want the exact name)
    # We check if it's numeric, but cardinality can run on keywords too.
    target_field = field
    info = find_field_info(field, inventory)
    if not info:
        # Try to find it in inventory by name match
        info = next((f for f in inventory if f["name"] == field), None)
    
    if info:
        target_field = info["name"]
        if metric != "cardinality" and not info.get("isNumeric"):
             return {"error": f"Field '{field}' is not numeric. Cannot calculate '{metric}'."}

    valid_metrics = {"avg", "max", "min", "sum", "cardinality", "value_count"}
    if metric not in valid_metrics:
        return {"error": f"Invalid metric '{metric}'. Must be one of {valid_metrics}"}

    agg_def = {metric: {"field": target_field}}
    
    aggs = {}
    if interval:
        aggs["trend"] = {
            "date_histogram": {
                "field": time_field,
                "fixed_interval": interval,
                "min_doc_count": 0
            },
            "aggs": {"metric_value": agg_def}
        }
    else:
        aggs["overall_metric"] = agg_def

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": build_time_filters(lookback, time_field)
            }
        },
        "aggs": aggs,
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to calculate metrics: {exc}"}

    result = {
        "metric": metric,
        "field": target_field,
        "time_window": {"gte": lookback, "lte": "now"},
    }

    if interval:
        buckets = response.get("aggregations", {}).get("trend", {}).get("buckets", [])
        trend_data = []
        for bucket in buckets:
            val = bucket.get("metric_value", {}).get("value")
            trend_data.append({
                "timestamp": bucket.get("key_as_string"),
                "value": val
            })
        result["trend"] = trend_data
        result["interval"] = interval
    else:
        val = response.get("aggregations", {}).get("overall_metric", {}).get("value")
        result["value"] = val

    return result


@server.tool()
def get_top_values(
    index: Union[str, List[str]],
    field: str,
    size: int = 10,
    lookback: str = "now-24h",
    field_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Find most frequent values for a field.

    Args:
        index: Index name/pattern.
        field: Field to aggregate.
        size: Number of results.
        lookback: Time window.
        field_overrides: Optional overrides.
    """
    indices = normalize_indices_param(index)
    if not indices:
        return {"error": "index must be provided"}

    client = get_es_client()
    inventory = extract_field_inventory_for_indices(indices)
    fields = resolve_log_fields(field_overrides)
    time_field = fields.get("time", DEFAULT_TIME_FIELD) or DEFAULT_TIME_FIELD

    # Resolve field to keyword if possible
    target_field = ensure_terms_field(field, inventory) or field

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": build_time_filters(lookback, time_field)
            }
        },
        "aggs": {
            "top_values": {
                "terms": {
                    "field": target_field,
                    "size": size,
                    "missing": "UNKNOWN"
                }
            }
        },
        "timeout": f"{SEARCH_TIMEOUT_MS}ms",
    }

    try:
        response = client.search(index=indices, body=body)
    except es_exceptions.ElasticsearchException as exc:
        return {"error": f"Failed to get top values: {exc}"}

    buckets = response.get("aggregations", {}).get("top_values", {}).get("buckets", [])
    values = parse_terms_buckets(buckets)

    return {
        "field": target_field,
        "time_window": {"gte": lookback, "lte": "now"},
        "top_values": values,
    }


if ENABLE_PLANNER:
    @server.tool()
    def plan_query(
        nl: str,
        indices: Optional[List[str]] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translates a natural language question into an Elasticsearch Query DSL.

        This tool uses an LLM (if configured) or local heuristics to convert a user's
        question (e.g., "show me login errors from the last hour") into a valid
        Elasticsearch DSL query. The result includes the generated DSL, the target
        indices, and the planner's confidence and assumptions.

        Args:
            nl: The natural language question to be translated into a query.
            indices: Optional list of index names to search. If not provided, the
                planner will attempt to select relevant indices.
            hints: Optional dictionary of hints to guide the planning process.
        """

        if not nl:
            return {"error": "nl must be provided."}
        plan = plan_query_internal(nl, indices)
        return plan


# -----------------------------------------------------------------------------
# Server start
# -----------------------------------------------------------------------------


def log_startup() -> None:
    """Log configuration at startup."""

    tools_enabled = [
        "summarize_logs",
        "log_trend",
        "sample_trace",
        "list_indices",
        "get_mapping",
        "get_field_caps",
        "sample_values",
        "execute_search",
        "search_with_projection",
        "count_and_aggregate",
        "search_paginated",
        "estimate_response_size",
        "sample_logs_stratified",
        "get_cluster_health",
        "find_traces_by_user",
        "analyze_error_patterns",
        "get_metric_statistics",
        "get_top_values",
    ]
    if ENABLE_PLANNER:
        tools_enabled.append("plan_query")
    print(
        "[mcp-elastic] Starting with ES URL="
        f"{ES_URL} api_key={'yes' if ES_API_KEY else 'no'} tools={','.join(tools_enabled)}"
    )
    print("[mcp-elastic] Typical flow: list_indices -> get_mapping -> execute_search")
    print("[mcp-elastic] Phase 1: count_and_aggregate (stats only) or search_with_projection (field filtering)")
    print("[mcp-elastic] Phase 2: search_paginated (large datasets), estimate_response_size (safety check), sample_logs_stratified (balanced sampling)")
    print("[mcp-elastic] Enhanced summarize_logs: configurable sampling (size, strategy, message truncation)")
    if ENABLE_PLANNER:
        print("[mcp-elastic] Planner flow: plan_query -> execute_search")


if __name__ == "__main__":
    log_startup()
    server.run(transport="streamable-http")
    #server.run(transport="stdio")
