"""
Cypher query builder — generates parameterized Neo4j Cypher queries.

All Cypher string construction is isolated here.
Using parameterized queries prevents injection vulnerabilities.
"""

from __future__ import annotations


class CypherBuilder:
    """Builds parameterized Cypher queries for entity and relation operations."""

    @staticmethod
    def upsert_entity(entity_type: str) -> str:
        """
        Build a MERGE query to upsert an entity node.

        Args:
            entity_type: The node label (e.g. "LegalConcept", "Pasal").

        Returns:
            Parameterized Cypher string. Parameters: {name, properties}.
        """
        clean_type = entity_type.replace(" ", "_").replace("-", "_")
        return (
            f"MERGE (n:{clean_type} {{canonical_name: $name}}) "
            f"ON CREATE SET n += $properties, n.created_at = datetime() "
            f"ON MATCH SET n += $properties, n.updated_at = datetime() "
            f"RETURN n"
        )

    @staticmethod
    def upsert_relation() -> str:
        """
        Build a MERGE query to upsert a relation between two entities.

        Returns:
            Parameterized Cypher. Parameters: {from_name, to_name, rel_type, properties}.
        """
        
        return (
            "MATCH (a {canonical_name: $from_name}) "
            "MATCH (b {canonical_name: $to_name}) "
                "CALL apoc.merge.relationship(a, $rel_type, {}, $properties, b) "
                "YIELD rel RETURN rel"
        )

    @staticmethod
    def multi_hop_path(max_hops: int) -> str:
        """
        Build a variable-length path query from a starting entity.

        Args:
            max_hops: Maximum traversal depth.

        Returns:
            Parameterized Cypher. Parameters: {start_name}.
        """
        return (
            f"MATCH path = (start {{canonical_name: $start_name}})"
            f"-[*1..{max_hops}]-(related) "
            f"RETURN nodes(path) as nodes, relationships(path) as rels, "
            f"length(path) as path_length "
            f"LIMIT 50"
        )
