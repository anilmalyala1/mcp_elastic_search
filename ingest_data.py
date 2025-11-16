"""
This script generates and ingests random log data into an Elasticsearch data stream.

The script creates a specified number of log records with a predefined structure
and uses the Elasticsearch bulk helper for efficient ingestion.
"""

import os
import random
import uuid
from datetime import datetime, UTC

from elasticsearch import NotFoundError
from elasticsearch.helpers import bulk
from faker import Faker

from es_client import create_elasticsearch_client

# Configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
DATA_STREAM_NAME = "webapplication-logs"
INDEX_TEMPLATE_NAME = "webapplication-logs-template"
INDEX_PATTERN = "webapplication-logs*"
NUM_RECORDS = 100000

# Initialize Faker for generating random data
fake = Faker()


def create_index_template(es_client):
    """Creates an index template for the logs data stream."""
    template = {
        "index_patterns": [INDEX_PATTERN],
        "data_stream": {},
        "template": {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "log_message": {"type": "text"},
                    "service_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "response_status": {"type": "integer"},
                    "response_time_ms": {"type": "integer"},
                    "path": {"type": "keyword"},
                }
            }
        },
    }
    es_client.indices.put_index_template(name=INDEX_TEMPLATE_NAME, body=template)

def get_log_message_template(service_id, status_code):
    """
    Returns a realistic log message template based on the service and status code.
    Use placeholders like {user_id}, {path}, {span_id}
    """
    
    # --- INFO/SUCCESS MESSAGES (200, 201) ---
    if status_code in [200, 201]:
        templates = {
            "auth-service": [
                "User {user_id} logged in successfully.",
                "Session refreshed for user {user_id}. Path: {path}",
                "User {user_id} token successfully validated.",
            ],
            "user-service": [
                "User {user_id} profile retrieved.",
                "New user account created for {user_id}.",
                "User {user_id} updated email address.",
            ],
            "payment-service": [
                "Payment transaction initiated for user {user_id}. Txn ID: {span_id}.",
                "Subscription renewed for user {user_id}.",
                "Billing details updated for {user_id}.",
            ],
            "order-service": [
                "Order {span_id} successfully created by user {user_id}.",
                "Order items retrieved for order {span_id}.",
                "Order status updated to 'Processing' for {span_id}.",
            ],
        }
    
    # --- CLIENT ERROR MESSAGES (400, 404) ---
    elif status_code in [400, 404]:
        templates = {
            "auth-service": [
                "Login failed for user {user_id}: Invalid credentials.",
                "User session expired for {user_id}.",
                "Authentication required for path: {path}",
            ],
            "user-service": [
                "Resource not found at path: {path}",
                "Validation error: Missing required field for user {user_id}.",
                "Attempt to access non-existent user profile {user_id}.",
            ],
            "payment-service": [
                "Payment validation failed for user {user_id}: Invalid card details.",
                "400 Bad Request: Malformed request body for {path}.",
                "Transaction declined for user {user_id}.",
            ],
            "order-service": [
                "Order creation failed for {user_id}: Cart is empty.",
                "Requested order ID {span_id} not found.",
                "404 Not Found: Product list access failed at {path}.",
            ],
        }

    # --- SERVER ERROR MESSAGES (500) ---
    elif status_code == 500:
        templates = {
            "auth-service": [
                "500 Internal Server Error: Database connection failed during login for {user_id}.",
                "Critical error in token generation service.",
            ],
            "user-service": [
                "500 Internal Server Error: Unhandled exception in user data retrieval.",
                "Failed to write user profile to cache for {user_id}.",
            ],
            "payment-service": [
                "Gateway timeout during payment processing. Txn ID: {span_id}.",
                "CRITICAL: Third-party payment API failed.",
            ],
            "order-service": [
                "500 Internal Server Error: Failed to commit order transaction {span_id}.",
                "Dependency service unresponsive during order creation.",
            ],
        }

    # Fallback to a generic error message
    else:
        return f"Generic request handled for {service_id} on {path}"
    
    # Return a randomly chosen message from the category for the given service
    return random.choice(templates.get(service_id, ["Unknown operation occurred."]))

def create_random_log_record():
    """Creates a single random log record with a realistic log message."""
    
    # 1. Generate core fields first
    service_id = random.choice(
        ["auth-service", "user-service", "payment-service", "order-service"]
    )
    user_id = fake.user_name()
    response_status = random.choice([200, 201, 400, 404, 500])
    path = fake.uri_path()
    span_id = str(uuid.uuid4())[:8] # Used as Transaction/Order ID sometimes
    
    # 2. Get the template and format the message
    template = get_log_message_template(service_id, response_status)
    log_message = template.format(
        user_id=user_id, 
        path=path, 
        span_id=span_id
    )

    # 3. Construct the final log record
    return {
        "_op_type": "create",
        "_index": DATA_STREAM_NAME,
        "_source": {
            "trace_id": str(uuid.uuid4()),
            "span_id": span_id, # Use the generated span_id
            "log_message": log_message,
            "service_id": service_id,
            "user_id": user_id,
            "response_status": response_status,
            "response_time_ms": random.randint(50, 2000),
            "path": path,
            "@timestamp": datetime.now(UTC).isoformat(),
        },
    }


def generate_logs(num_logs):
    """Generator function to yield log records."""
    for _ in range(num_logs):
        yield create_random_log_record()


def main():
    """Main function to generate and ingest data."""
    print(f"Connecting to Elasticsearch at {ELASTICSEARCH_URL}...")
    es_client = create_elasticsearch_client(ELASTICSEARCH_URL, api_key=API_KEY)

    if not es_client.indices.exists_index_template(name=INDEX_TEMPLATE_NAME):
        print(f"Creating index template '{INDEX_TEMPLATE_NAME}'...")
        create_index_template(es_client)

    try:
        es_client.indices.get_data_stream(name=DATA_STREAM_NAME)
        print(f"Data stream '{DATA_STREAM_NAME}' already exists.")
    except Exception as e:
        print(f"Creating data stream '{DATA_STREAM_NAME}'...")
        es_client.indices.create_data_stream(name=DATA_STREAM_NAME)

    print(f"Generating and ingesting {NUM_RECORDS} log records...")

    successes, failed_actions = bulk(
        es_client,
        generate_logs(NUM_RECORDS),
        chunk_size=1000,
        request_timeout=60,
        max_retries=5,
    )

    print(f"Successfully ingested {successes}/{NUM_RECORDS} records.")
    if failed_actions:
        print(f"Failed to ingest {len(failed_actions)} documents:")
        for item in failed_actions:
            print(item)


if __name__ == "__main__":
    main()
