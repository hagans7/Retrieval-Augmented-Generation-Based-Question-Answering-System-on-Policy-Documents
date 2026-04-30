"""
Neo4j graph client — entry point implementing BaseGraphClient.
"""

from __future__ import annotations

from neo4j import AsyncGraphDatabase, AsyncDriver

from src.clients.graph.cypher_builder import CypherBuilder
from src.clients.graph.result_mapper import GraphResultMapper
from src.core.exceptions.infrastructure import GraphQueryError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_graph_client import BaseGraphClient


class Neo4jGraphClient(BaseGraphClient):
    """
    Async Neo4j graph client.

    Args:
        uri: Bolt connection URI (e.g. "bolt://neo4j:7687").
        user: Neo4j username.
        password: Neo4j password.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None
        self._logger = get_logger(__name__)

    async def connect(self) -> None:
        """Open the Neo4j async driver. Called at application startup."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._driver.verify_connectivity()
        self._logger.info("Neo4j connection established", extra={"uri": self._uri})

    async def close(self) -> None:
        """Close the Neo4j driver at application shutdown."""
        if self._driver:
            await self._driver.close()

    async def upsert_entities(self, entities: list[dict]) -> None:
        """
        Upsert entities into the graph.

        Each dict must have: entity_type, canonical_name, and optional properties.

        Raises:
            GraphQueryError: If the operation fails.
        """
        if not entities or not self._driver:
            return
        try:
            async with self._driver.session() as session:
                async with await session.begin_transaction() as tx:
                    for entity in entities:
                        query = CypherBuilder.upsert_entity(entity["entity_type"])
                        await tx.run(
                            query,
                            name=entity["canonical_name"],
                            properties={
                                k: v for k, v in entity.items()
                                if k not in ("entity_type", "canonical_name")
                            },
                        )
                    await tx.commit()
        except Exception as exc:
            raise GraphQueryError(
                message="Failed to upsert entities.",
                context={"count": len(entities), "error": str(exc)},
            ) from exc

        
    # async def upsert_relations(self, relations: list[dict]) -> None:
    #     """
    #     Upsert relations between entities.

    #     Each dict must have: from_entity, to_entity, relation_type.

    #     Raises:
    #         GraphQueryError: If the operation fails.
    #     """
    #     if not relations or not self._driver:
    #         return
    #     try:
    #         async with self._driver.session() as session:
    #             async with await session.begin_transaction() as tx:
    #                 for rel in relations:
    #                     query = CypherBuilder.upsert_relation()
    #                     await tx.run(
    #                         query,
    #                         from_name=rel["from_entity"],
    #                         to_name=rel["to_entity"],
    #                         rel_type=rel["relation_type"],
    #                         properties={
    #                             k: v for k, v in rel.items()
    #                             if k not in ("from_entity", "to_entity", "relation_type")
    #                         },
    #                     )
    #                 await tx.commit()
    #     except Exception as exc:
    #         raise GraphQueryError(
    #             message="Failed to upsert relations.",
    #             context={"count": len(relations), "error": str(exc)},
    #         ) from exc

    async def upsert_relations(self, relations: list[dict]) -> None:
        """
        Upsert relations between entities.

        Each dict must have: from_entity, to_entity, relation_type.

        Raises:
            GraphQueryError: If the operation fails.
        """
        if not relations or not self._driver:
            return
        try:
            async with self._driver.session() as session:
                async with await session.begin_transaction() as tx:
                    for rel in relations:
                        query = CypherBuilder.upsert_relation()
                        
                        # Sanitize relation type: uppercase and underscores
                        raw_rel_type = rel["relation_type"]
                        clean_rel_type = raw_rel_type.replace(" ", "_").replace("-", "_").upper()
                        
                        await tx.run(
                            query,
                            from_name=rel["from_entity"],
                            to_name=rel["to_entity"],
                            rel_type=clean_rel_type,
                            properties={
                                k: v for k, v in rel.items()
                                if k not in ("from_entity", "to_entity", "relation_type")
                            },
                        )
                    await tx.commit()
        except Exception as exc:
            raise GraphQueryError(
                message="Failed to upsert relations.",
                context={"count": len(relations), "error": str(exc)},
            ) from exc

    async def multi_hop_query(
        self,
        start_entity: str,
        max_hops: int,
    ) -> list[dict]:
        """
        Traverse graph from a starting entity.

        Returns:
            List of path result dicts.

        Raises:
            GraphQueryError: If the query fails.
        """
        if not self._driver:
            return []
        try:
            query = CypherBuilder.multi_hop_path(max_hops)
            async with self._driver.session() as session:
                result = await session.run(query, start_name=start_entity)
                records = await result.data()
            return GraphResultMapper.map_path_results(records)
        except Exception as exc:
            raise GraphQueryError(
                message=f"Multi-hop query from '{start_entity}' failed.",
                context={"start_entity": start_entity, "max_hops": max_hops, "error": str(exc)},
            ) from exc
