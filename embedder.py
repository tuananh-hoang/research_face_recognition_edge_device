"""
Module A — Feature Extraction
Dùng ArcFace (InsightFace) pretrained làm backbone cố định.
Output: embedding 512-d, L2-normalized, bbox, det_score
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_sc", providers=None):
        """
        buffalo_sc = small model (~300KB), CPU-friendly
        buffalo_l  = large model, more accurate
        """
        from insightface.app import FaceAnalysis
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.embed_dim = 512

    def get_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Args:
            image: BGR image (OpenCV format)
        Returns:
            emb: L2-normalized 512-d embedding, hoặc None nếu không detect được mặt
            bbox: [x1, y1, x2, y2]
            det_score: detection confidence [0,1]
        """
        if image is None or image.size == 0:
            return None, None, 0.0

        faces = self.app.get(image)
        if not faces:
            return None, None, 0.0

        # Chọn mặt có det_score cao nhất
        face = max(faces, key=lambda x: x.det_score)

        raw_emb = face.embedding  # shape (512,)
        norm = np.linalg.norm(raw_emb)

        # Lưu raw_norm trước khi normalize (dùng cho IQA quality proxy)
        self._last_raw_norm = float(norm)

        # L2 normalize
        emb = raw_emb / (norm + 1e-8)

        return emb, face.bbox, float(face.det_score)

    def get_raw_norm(self) -> float:
        """Trả về feature norm trước khi normalize (proxy cho image quality)"""
        return getattr(self, '_last_raw_norm', 0.0)


# ─── Sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    embedder = FaceEmbedder()

    # Tạo ảnh test dummy
    dummy = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    emb, bbox, score = embedder.get_embedding(dummy)

    if emb is not None:
        print(f"Embedding shape : {emb.shape}")
        print(f"L2 norm (should ≈ 1.0): {np.linalg.norm(emb):.4f}")
        print(f"Det score : {score:.3f}")
        print(f"Raw feature norm : {embedder.get_raw_norm():.2f}")
    else:
        print("No face detected in dummy image (expected for random pixels)")
    print("Module A OK")