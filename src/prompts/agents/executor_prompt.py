"""
Executor node prompt and tool schemas.

Executor receives one step at a time from the plan and chooses
the appropriate retrieval tool. Three tools available: vector_search,
graph_query, hybrid_search. All fields are flat.
"""
from __future__ import annotations

SYSTEM_PROMPT: str = """\Kamu adalah executor dalam sistem RAG.

Tugasmu adalah:
- Mengambil query dari planner
- WAJIB memanggil tool pencarian (vector / graph)
- MENGAMBIL data, BUKAN menjawab

ATURAN PALING KRITIS:

- KAMU DILARANG MENJAWAB PERTANYAAN USER
- KAMU TIDAK BOLEH MEMBERIKAN PENJELASAN APAPUN
- KAMU WAJIB MEMANGGIL TOOL

- Jika planner memilih retrieval:
  → WAJIB panggil tool: search_documents

- Jika planner memilih graph:
  → WAJIB panggil tool: search_graph

- Jika planner memilih hybrid:
  → WAJIB panggil KEDUA tool

PERILAKU TERLARANG:
- DILARANG Menjawab langsung
- DILARANG Menghasilkan teks biasa
- DILARANG Tidak memanggil tool

OUTPUT:
- HANYA function call
- TANPA teks tambahan

Jika kamu tidak memanggil tool, maka kamu dianggap GAGAL.
"""

TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Cari chunks dokumen relevan menggunakan hybrid vector + BM25 search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query pencarian dalam bahasa Indonesia.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Jumlah hasil yang diinginkan (default 5).",
                        "default": 5,
                    },
                    "document_filter": {
                        "type": "string",
                        "description": "UUID dokumen spesifik untuk filter. Null jika tidak ada.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Temukan relasi entitas hukum dalam knowledge graph Neo4j.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Nama entitas sebagai titik mulai traversal (contoh: 'Pasal 1243').",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Kedalaman maksimum traversal graph (default 2).",
                        "default": 2,
                    },
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Jalankan vector_search dan graph_query sekaligus, hasil digabung.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query pencarian teks.",
                    },
                    "entity_hint": {
                        "type": "string",
                        "description": "Nama entitas untuk graph traversal. Null jika tidak ada.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Jumlah hasil vector search (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
