"""
Graph result mapper — maps Neo4j records to standard dicts.

All neo4j.Record and Node/Relationship types are consumed here.
Service layer receives clean Python dicts.
"""

from __future__ import annotations


class GraphResultMapper:
    """Maps Neo4j query records to standardized Python dicts."""

    @staticmethod
    def map_entities(records: list) -> list[dict]:
        """
        Map upserted entity records to dicts.

        Args:
            records: Neo4j query result records.

        Returns:
            List of entity dicts with node properties.
        """
        results = []
        for record in records:
            node = record.get("n")
            if node:
                results.append(dict(node))
        return results

    @staticmethod
    def map_path_results(records: list) -> list[dict]:
        """
        Map multi-hop path query records to structured dicts.

        Args:
            records: Neo4j path query result records.

        Returns:
            List of dicts with nodes, relationships, and path_length.
        """
        results = []
        for record in records:
            nodes = [dict(n) for n in (record.get("nodes") or [])]
            rels = []
            for rel in (record.get("rels") or []):
                rels.append({
                    "type": rel.type,
                    "properties": dict(rel),
                    "start": rel.start_node.element_id,
                    "end": rel.end_node.element_id,
                })
            results.append({
                "nodes": nodes,
                "relations": rels,
                "path_length": record.get("path_length", 0),
            })
        return results
