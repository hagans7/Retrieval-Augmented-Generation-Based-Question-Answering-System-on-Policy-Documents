"""
OCR model loader — DEPRECATED / NOT USED in current architecture.

Retained for reference and potential future use if a custom OCR model
needs to be integrated into Docling as a backend plugin.

Current architecture:
    Docling's DocumentConverter handles OCR natively via its own pipeline.
    No external model loading is required at startup or at runtime.

If LightOnOCR-2-1B needs to be integrated in the future:
    1. Wait for transformers >= 4.50.0 (mistral3 architecture support).
    2. Use Docling's RapidOcrOptions or a custom OCR backend class.
    3. Inject via DocumentConverter(pipeline_options=...).

This file is intentionally left as a placeholder — do not import
or instantiate OCRModelLoader in the current production pipeline.
"""

from __future__ import annotations

from pathlib import Path

from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class OCRModelLoader:
    """
    Placeholder for future custom OCR model loading.

    NOT used in the current production pipeline.
    Docling's native OCR backend is used instead.
    """

    def __init__(
        self,
        model_repo: str,
        model_dir: str,
        hf_token: str | None = None,
    ) -> None:
        self._model_repo = model_repo
        self._model_dir = Path(model_dir)
        self._hf_token = hf_token
        logger.debug(
            "OCRModelLoader instantiated but not active — Docling native pipeline in use.",
            extra={"model_repo": model_repo},
        )