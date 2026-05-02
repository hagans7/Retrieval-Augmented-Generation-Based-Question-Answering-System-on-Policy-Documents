"""
Graph result mapper — maps Neo4j records to standard Python dicts.

All Neo4j driver types (Record, Node, Relationship) are consumed here.
Service layer receives only clean, JSON-serializable Python dicts.

Format contract from CypherBuilder.multi_hop_path:
  - records[*]["nodes"]        → list[dict]  (node properties via properties(n))
  - records[*]["rels"]         → list[dict]  with keys:
      type:        str
      start_id:    str  (stringified node id)
      end_id:      str
      properties:  dict
  - records[*]["path_length"]  → int

Because the Cypher query projects nodes and relationships as plain maps
(using properties(n) and inline map projection for relationships),
result.data() returns pure Python dicts — no Relationship or Node objects
that would cause AttributeError on .type, .start_node, etc.

Temporal type sanitization:
  Neo4j node properties may contain temporal types such as neo4j.time.DateTime,
  neo4j.time.Date, neo4j.time.Time, neo4j.time.Duration, and spatial types
  like neo4j.spatial.Point. These are not JSON-serializable.

  _sanitize_value() converts all such types to their ISO 8601 string representation
  so that the returned dicts can safely be passed to json.dumps() anywhere in the
  service layer — specifically in agent_runner._run_graph_query().

  This is the correct place for this conversion: the mapper is the only layer
  that should know about Neo4j-specific types. All layers above it deal only
  with plain Python dicts.
"""
from __future__ import annotations


class GraphResultMapper:
    """Maps Neo4j query records to standardized, JSON-safe Python dicts."""

    @staticmethod
    def _sanitize_value(value: object) -> object:
        """
        Convert Neo4j-specific types to JSON-serializable Python primitives.

        Handles:
          - neo4j.time.DateTime / Date / Time / LocalDateTime / LocalTime
            → ISO 8601 string via .iso_format() or str()
          - neo4j.time.Duration → str()
          - neo4j.spatial.Point → {"x": ..., "y": ..., "z": ...} dict
          - list / dict → recursively sanitized
          - All other types → returned as-is (already JSON-safe)

        This is called on every value inside node and relationship property
        dicts so that callers (agent_runner) can safely call json.dumps()
        without needing a custom default= encoder.
        """
        type_name = type(value).__name__
        module = getattr(type(value), "__module__", "")

        # Neo4j temporal types (neo4j.time module)
        if "neo4j" in module and "time" in module:
            if hasattr(value, "iso_format"):
                return value.iso_format()
            return str(value)

        # Neo4j spatial types (neo4j.spatial module)
        if "neo4j" in module and "spatial" in module:
            result: dict = {}
            for attr in ("x", "y", "z", "longitude", "latitude", "height"):
                if hasattr(value, attr):
                    result[attr] = getattr(value, attr)
            return result if result else str(value)

        # Recurse into containers
        if isinstance(value, dict):
            return {k: GraphResultMapper._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GraphResultMapper._sanitize_value(v) for v in value]

        return value

    @staticmethod
    def _sanitize_dict(d: dict) -> dict:
        """Recursively sanitize all values in a dict."""
        return {k: GraphResultMapper._sanitize_value(v) for k, v in d.items()}

    @staticmethod
    def map_entities(records: list) -> list[dict]:
        """
        Map upserted entity records to dicts.

        Args:
            records: Neo4j query result records (from result.data()).

        Returns:
            List of entity dicts with node properties, all JSON-serializable.
        """
        results = []
        for record in records:
            node = record.get("n")
            if node:
                raw = dict(node)
                results.append(GraphResultMapper._sanitize_dict(raw))
        return results

    @staticmethod
    def map_path_results(records: list) -> list[dict]:
        """
        Map multi-hop path query records to structured dicts.

        Expects records produced by CypherBuilder.multi_hop_path which projects:
          - nodes as properties(n)  → list[dict]
          - rels as inline map      → list[dict] with type, start_id, end_id, properties
          - path_length             → int

        This approach is safe regardless of Neo4j driver version because
        nodes and rels are already plain Python dicts from the Cypher projection.

        Args:
            records: List of dicts from result.data().

        Returns:
            List of path result dicts.
        """
        results = []
        for record in records:
            # nodes: already list[dict] from properties(n), but may contain
            # Neo4j temporal types — sanitize each property value
            raw_nodes = record.get("nodes") or []
            nodes = [
                GraphResultMapper._sanitize_dict(dict(n) if not isinstance(n, dict) else n)
                for n in raw_nodes
            ]

            # rels: already list[dict] from inline Cypher map projection
            raw_rels = record.get("rels") or []
            rels = []
            for rel in raw_rels:
                if isinstance(rel, dict):
                    # Expected format: {type, start_id, end_id, properties}
                    raw_props = rel.get("properties", {})
                    rels.append({
                        "type": rel.get("type", ""),
                        "start_id": rel.get("start_id", ""),
                        "end_id": rel.get("end_id", ""),
                        "properties": GraphResultMapper._sanitize_dict(
                            raw_props if isinstance(raw_props, dict) else {}
                        ),
                    })
                else:
                    # Fallback for unexpected types — extract what we can
                    try:
                        rels.append({
                            "type": getattr(rel, "type", str(type(rel))),
                            "start_id": "",
                            "end_id": "",
                            "properties": dict(rel) if hasattr(rel, "__iter__") else {},
                        })
                    except Exception:
                        continue

            results.append({
                "nodes": nodes,
                "relations": rels,
                "path_length": record.get("path_length", 0),
            })
        return results