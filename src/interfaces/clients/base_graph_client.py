"""Abstract base for all graph database clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGraphClient(ABC):
    """Contract for all graph database clients (Neo4j, etc.)."""

    @abstractmethod
    async def upsert_entities(self, entities: list[dict]) -> None:
        """
        Insert or update entities in the graph database.

        Each entity dict must contain: entity_type, canonical_name,
        and optional properties.

        Args:
            entities: List of entity dicts to upsert.

        Raises:
            GraphQueryError: If the upsert fails.
        """
        ...

    @abstractmethod
    async def upsert_relations(self, relations: list[dict]) -> None:
        """
        Insert or update relations between entities.

        Each relation dict must contain: from_entity, to_entity,
        relation_type, and optional properties.

        Args:
            relations: List of relation dicts to upsert.

        Raises:
            GraphQueryError: If the upsert fails.
        """
        ...

    @abstractmethod
    async def multi_hop_query(
        self,
        start_entity: str,
        max_hops: int,
    ) -> list[dict]:
        """
        Traverse the graph from a starting entity up to max_hops away.

        Args:
            start_entity: Canonical name of the starting entity node.
            max_hops: Maximum traversal depth.

        Returns:
            List of dicts representing found nodes and their paths.

        Raises:
            GraphQueryError: If the query fails.
        """
        ...
