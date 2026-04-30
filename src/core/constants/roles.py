"""
Role and status constants for all domain enums.

These constants are the single source of truth for string values
used in enums across ORM models, entities, and business logic.
Never use string literals for roles or statuses — always reference here.
"""

# Message roles — stored in messages.role column
MESSAGE_ROLE_USER: str = "user"
MESSAGE_ROLE_ASSISTANT: str = "assistant"

# User roles — stored in users.role column
USER_ROLE_USER: str = "user"
USER_ROLE_ADMIN: str = "admin"
USER_ROLE_SUPERVISOR: str = "supervisor"

# Document ingestion statuses — stored in documents.ingestion_status column
INGESTION_STATUS_PENDING: str = "pending"
INGESTION_STATUS_PROCESSING: str = "processing"
INGESTION_STATUS_COMPLETED: str = "completed"
INGESTION_STATUS_FAILED: str = "failed"
