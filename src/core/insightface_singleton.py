"""
InsightFace singleton cho toàn bộ project.

Dùng chung một instance InsightFace trong toàn bộ session,
tránh khởi tạo lại model nhiều lần.
"""
from __future__ import annotations

import numpy as np
import cv2

__all__ = ['InsightFaceSingleton']


class InsightFaceSingleton:
    """
    Lazy-loading singleton cho InsightFace buffalo_sc model.

    Sử dụng:
        app = InsightFaceSingleton.get_instance()
        faces = app.get(image)

    Nếu insightface không có sẵn → tự động dùng mock mode.
    """
    _instance = None
    _model_name = 'buffalo_sc'
    _providers = ['CPUExecutionProvider']

    @classmethod
    def get_instance(cls):
        """Lấy (hoặc tạo mới nếu chưa có) instance InsightFace."""
        if cls._instance is None:
            cls._instance = cls._create_instance()
        return cls._instance

    @classmethod
    def _create_instance(cls):
        """Khởi tạo InsightFace hoặc mock fallback."""
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name=cls._model_name,
                providers=cls._providers,
            )
            app.prepare(ctx_id=0, det_size=(320, 320))
            print(f"[InsightFace] Loaded buffalo_sc model (CPU)")
            return app

        except ImportError:
            print("\n[WARNING] insightface not found — using mock embeddings")
            print("  Install with: pip install insightface")
            return _MockInsightFaceApp()

        except Exception as exc:
            print(f"\n[WARNING] InsightFace init failed ({exc}) — using mock embeddings")
            print("  Model file may be missing. Run: python -c 'from insightface.app import FaceAnalysis; "
                  "FaceAnalysis(name=\"buffalo_sc\").prepare(0)' to download.")
            return _MockInsightFaceApp()

    @classmethod
    def reset(cls):
        """Reset instance (chủ yếu dùng cho testing)."""
        cls._instance = None

    @classmethod
    def is_mock(cls) -> bool:
        """True nếu đang dùng mock mode."""
        return isinstance(cls.get_instance(), _MockInsightFaceApp)


# ── Mock fallback ──────────────────────────────────────────────────────────────


class _MockFace:
    """Mock face object trả về từ mock app."""
    def __init__(self, emb: np.ndarray, det_score: float):
        self.embedding = emb
        self.det_score = det_score


class _MockInsightFaceApp:
    """
    Mock InsightFace app — dùng khi không cài đặt insightface.

    Trả về embeddings từ resized images (L2-normalized).
    Chỉ dùng cho development/testing, KHÔNG dùng cho production.
    """
    def get(self, image) -> list[_MockFace]:
        """
        Trả về list of MockFace từ image.

        Fallback strategy:
          1. Resize ảnh về 112x112 (InsightFace standard size)
          2. Flatten thành vector
          3. L2-normalize
          4. Gán det_score ngẫu nhiên 0.5–0.99
        """
        img_resized = cv2.resize(image, (112, 112))
        vec = img_resized.flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec) + 1e-8)
        emb = vec / norm

        det_score = float(np.random.uniform(0.50, 0.99))

        return [_MockFace(emb, det_score)]
