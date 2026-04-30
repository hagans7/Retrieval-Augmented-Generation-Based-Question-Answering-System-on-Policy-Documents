"""
Domain and business logic exception variants.

These represent failures within application domain logic — persistence,
agent orchestration, and validation failures.
"""

from __future__ import annotations

from src.core.exceptions.base import AppBaseError


class PersistenceError(AppBaseError):
    """
    Raised when a database read or write operation fails unexpectedly.

    Critical — workflow cannot continue if persistence fails.
    HTTP mapping: 503 Service Unavailable.
    Raised by: all repository files.
    """


class DuplicateResourceError(PersistenceError):
    """
    Raised when an INSERT violates a unique constraint.

    Subclass of PersistenceError for targeted handling when needed.
    HTTP mapping: 409 Conflict (handled explicitly) or 503 (default).

    Args:
        resource_type: The type of resource that already exists.
        identifier: The conflicting identifier.
    """

    def __init__(self, resource_type: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource_type} with identifier '{identifier}' already exists.",
            context={"resource_type": resource_type, "identifier": identifier},
        )


class AgentExecutionError(AppBaseError):
    """
    Raised when the LangGraph agentic orchestrator encounters a fatal error.

    HTTP mapping: 503 Service Unavailable.
    Raised by: services/process_chat/agent_runner.py
    """


class PlanningError(AgentExecutionError):
    """
    Raised when the planner node fails to produce a valid plan.

    Subclass of AgentExecutionError.
    Raised by: services/process_chat/agent_runner.py (planner node)
    """


class AuditFailedError(AgentExecutionError):
    """
    Raised when the auditor node detects unresolvable evidence quality issues.

    Subclass of AgentExecutionError.
    Raised by: services/process_chat/agent_runner.py (auditor node)
    """


class ValidationError(AppBaseError):
    """
    Raised when input fails validation outside of Pydantic's scope.

    This is for business-rule validation, not schema validation.
    HTTP mapping: 422 Unprocessable Entity.

    Args:
        field: The field or parameter that failed validation.
        reason: Why validation failed.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            context={"field": field, "reason": reason},
        )


class ModelNotActiveError(ValidationError):
    """
    Raised when a user selects a model that exists but is not active.

    Subclass of ValidationError.
    HTTP mapping: 422 Unprocessable Entity.

    Args:
        model_id: The ID of the inactive model.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(
            field="model_id",
            reason=f"Model '{model_id}' exists but is not currently active.",
        )
