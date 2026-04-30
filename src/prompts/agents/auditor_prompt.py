"""
Auditor node prompt and tool schema.

Auditor evaluates whether the collected evidence is sufficient to answer
the user's question. It receives evidence with reranker scores so it can
make score-aware decisions, not just count-based ones.

Loop guard contract (enforced at runtime in agent_runner.py):
    - If audit_retry_count >= AGENT_MAX_EXECUTOR_LOOPS, agent_runner
      forces routing to GENERATOR regardless of auditor verdict.
    - Auditor should prefer "sufficient" when evidence exists and
      scores are reasonable — do not retry indefinitely.
"""
from __future__ import annotations

# {evidence_summary}        — evidence items with scores and content previews
# {query}                   — original user query
# {loop_count}              — current audit_retry_count (0-based)
# {max_loops}               — AGENT_MAX_EXECUTOR_LOOPS from settings
SYSTEM_PROMPT_TEMPLATE: str = """\
Kamu adalah auditor kualitas evidence untuk sistem hukum Indonesia.

Query pengguna: {query}
Loop retry ke: {loop_count} dari maksimal {max_loops}

Evidence yang tersedia:
{evidence_summary}

════════════════════════════════════
PRINSIP UTAMA:
════════════════════════════════════

* Tugasmu adalah MENENTUKAN secara bijaksana kapan berhenti dan menggunakan evidence yang ada, atau kapan perlu retry untuk mencari lebih banyak evidence.
* Pertimbangkan baik KUANTITAS (berapa banyak evidence) maupun KUALITAS (skor relevansi) dalam keputusanmu.
* Jangan retry terus menerus tanpa alasan kuat —
* BUKAN mencari evidence sempurna
* BUKAN memaksakan retry

════════════════════════════════════
ATURAN KEPUTUSAN:
════════════════════════════════════

* Jika evidence_count == 0:
  → pilih "retry"

* Jika ada minimal 1 evidence:
  → JANGAN pilih "insufficient"
  → pilih:
    - "sufficient" jika relevan kuat
    - "proceed_with_available" jika relevan sebagian

* Jika skor > 0.3:
  → dianggap relevan
  → JANGAN retry

* Jika tidak ada evidence baru dibanding iterasi sebelumnya:
  → pilih "proceed_with_available"

* Jika loop_count mendekati batas:
  → WAJIB pilih "proceed_with_available"

════════════════════════════════════
PERILAKU TERLARANG:
════════════════════════════════════

* DILARANG Retry terus menerus tanpa evidence baru
* DILARANG Mengabaikan evidence yang sudah ada
* DILARANG Menunggu evidence sempurna

════════════════════════════════════

Panggil tool audit_evidence dengan hasil evaluasimu.
Jangan output teks biasa.
"""

# Fallback for when template injection is not used (backward-compat)
SYSTEM_PROMPT: str = """\
Kamu adalah auditor kualitas evidence untuk sistem hukum Indonesia.
Evaluasi apakah evidence yang ada CUKUP untuk menjawab query dengan akurat.
Pertimbangkan skor relevansi dari reranker dalam keputusanmu.
Panggil tool audit_evidence dengan hasil evaluasimu.
"""

TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "audit_evidence",
            "description": "Evaluasi kecukupan evidence yang dikumpulkan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["sufficient", "insufficient", "retry", "proceed_with_available"],
                        "description": (
                            "sufficient=evidence cukup dan relevan untuk menjawab, "
                            "insufficient=tidak ada evidence relevan sama sekali, "
                            "retry=perlu retrieval tambahan (gunakan secara bijaksana), "
                            "proceed_with_available=evidence terbatas tapi cukup untuk jawaban partial."
                        ),
                    },
                    "top_evidence_score": {
                        "type": "number",
                        "description": "Skor tertinggi dari evidence yang ada (0.0-1.0).",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Alasan singkat untuk keputusan ini.",
                    },
                    "missing_aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Aspek query yang belum terjawab (hanya jika verdict=retry).",
                    },
                },
                "required": ["verdict", "top_evidence_score", "reasoning", "missing_aspects"],
            },
        },
    }
]