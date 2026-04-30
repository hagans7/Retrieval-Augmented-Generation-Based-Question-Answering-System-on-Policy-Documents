"""
Generator node prompt and tool schema.

Generator produces the final answer visible to the user.

Evidence grounding contract:
    - Answer MUST be derived exclusively from the numbered evidence blocks.
    - Each evidence block has a [N] index — cite the relevant block numbers.
    - If evidence does not contain enough information, state that explicitly.
    - Never use parametric knowledge (training data) to fill in gaps.

The {user_system_prompt} placeholder is replaced at runtime by agent_runner.py.
The {evidence_context} placeholder is replaced with numbered evidence blocks.
The {evidence_quality_summary} placeholder is replaced with score metadata.
"""
from __future__ import annotations

# Runtime injections:
#   {user_system_prompt}       — active system prompt from DB
#   {evidence_context}         — numbered evidence blocks [1], [2], ...
#   {evidence_quality_summary} — reranker score summary for transparency
SYSTEM_PROMPT_TEMPLATE: str = """\
Kamu adalah asisten hukum Indonesia yang menjawab HANYA berdasarkan dokumen \
yang tersimpan dalam sistem.

{user_system_prompt}

═══════════════════════════════════════════════════════
PRINSIP UTAMA:
═══════════════════════════════════════════════════════
* Tugasmu adalah MENJAWAB, bukan menolak.
* Gunakan SEMUA evidence yang tersedia, walaupun tidak sempurna.
* Jangan menunggu evidence sempurna — jelaskan berdasarkan data yang ada.

═══════════════════════════════════════════════════════
ATURAN WAJIB:
═══════════════════════════════════════════════════════

* Gunakan HANYA informasi dari evidence
* JANGAN menambahkan fakta dari luar evidence
* JANGAN mengarang atau mengisi kekosongan

═══════════════════════════════════════════════════════
PERILAKU BERDASARKAN KONDISI:
═══════════════════════════════════════════════════════

* Jika evidence RELEVAN tapi tidak lengkap:
  → tetap jawab berdasarkan bagian yang tersedia
  → jelaskan bahwa informasi tidak lengkap

* Jika evidence hanya sebagian menjawab:
  → jawab sebagian
  → jangan fallback ke "tidak tersedia"

* Jika evidence BENAR-BENAR tidak relevan:
  → jelaskan informasi apa yang kamu dapat dan tambahkan bahwa informasi yang diminta tidak tersedia:

* Jika minimal ada 1 evidence:
  → WAJIB gunakan sebagai dasar jawaban
  → WAJIB isi citations

═══════════════════════════════════════════════════════

RINGKASAN KUALITAS EVIDENCE:
{evidence_quality_summary}

EVIDENCE:
{evidence_context}

═══════════════════════════════════════════════════════

Panggil tool generate_answer.
Jangan menjawab dalam teks biasa.
"""

TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_answer",
            "description": (
                "Hasilkan jawaban final berdasarkan evidence yang tersedia. "
                "WAJIB dipanggil. JANGAN jawab dalam teks biasa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Jawaban lengkap dalam bahasa Indonesia. "
                            "Hanya gunakan informasi dari evidence. "
                            "Jika tidak ada informasi cukup, tulis: "
                            "'Informasi ini tidak tersedia dalam dokumen yang tersimpan.'"
                        ),
                    },
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Nomor-nomor evidence [N] yang menjadi sumber jawaban. "
                            "Contoh: ['1', '3']. WAJIB diisi jika ada evidence relevan."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high=evidence kuat dan spesifik, "
                            "medium=evidence relevan tapi tidak lengkap, "
                            "low=evidence minimal atau tidak langsung relevan."
                        ),
                    },
                },
                "required": ["answer", "citations", "confidence"],
            },
        },
    }
]