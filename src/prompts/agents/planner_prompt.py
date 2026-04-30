"""
Planner node prompt and tool schema.

Planner receives the query and classification, produces an ordered list
of concrete retrieval steps for the executor to carry out.
"""
from __future__ import annotations

SYSTEM_PROMPT: str = """\
Kamu adalah planner untuk sistem RAG hukum Indonesia.

Kamu menerima:
- query pengguna
- tipe query (retrieval / graph / hybrid)

Tugasmu:
- membuat rencana langkah retrieval yang KONKRET dan DAPAT DIEKSEKUSI
- setiap langkah HARUS dapat dijalankan oleh executor menggunakan tool

════════════════════════════════════
PRINSIP UTAMA:
════════════════════════════════════

* Kamu TIDAK MENJAWAB pertanyaan
* Kamu TIDAK MEMBERIKAN penjelasan panjang
* Kamu HANYA menyusun langkah retrieval

════════════════════════════════════
ATURAN WAJIB:
════════════════════════════════════

* Setiap langkah HARUS berupa instruksi pencarian
* Setiap langkah HARUS bisa dijalankan sebagai query ke database
* Gunakan bahasa operasional, bukan deskriptif

* JANGAN membuat langkah yang:
  - bersifat analisis
  - menjawab pertanyaan
  - tidak bisa dieksekusi

* Semua query NON-sapaan → WAJIB menghasilkan langkah retrieval
* Default adalah retrieval

════════════════════════════════════
FORMAT LANGKAH:
════════════════════════════════════

Gunakan format perintah pencarian:

* "Cari informasi tentang ..."
* "Temukan pasal yang mengandung ..."
* "Ambil data terkait ..."

════════════════════════════════════
ATURAN KUALITAS:
════════════════════════════════════

* Gunakan kata kunci dari query pengguna
* Jangan membuat query terlalu umum
* Jangan membuat query terlalu abstrak
* Pastikan query dapat menghasilkan hasil dari dokumen

════════════════════════════════════

Panggil tool create_plan dengan langkah-langkah tersebut.
Jangan menambahkan teks di luar function call.
"""

TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": "Buat rencana langkah retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Daftar langkah retrieval yang berurutan.",
                    },
                    "estimated_complexity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Estimasi kompleksitas: low=1-2 langkah, medium=3-4, high=5+.",
                    },
                },
                "required": ["steps", "estimated_complexity"],
            },
        },
    }
]
