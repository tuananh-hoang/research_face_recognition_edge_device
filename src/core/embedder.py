import sys

import cv2
import numpy as np

from src.core.model_config import get_face_model_config

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


class RealEmbedder:
    def __init__(self, model_name: str | None = None, det_size: tuple[int, int] | None = None):
        self.config = get_face_model_config(model_name, det_size)
        self.requested_model_name = self.config.requested_name
        self.model_name = self.config.model_name
        self.det_size = self.config.det_size
        self.model_description = self.config.description
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider'],
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            self.available = True
            print(
                f"[InsightFace] Loaded {self.model_name} "
                f"(requested={self.requested_model_name}, det_size={self.det_size})"
            )
            print(f"[InsightFace] {self.model_description}")
        except ImportError:
            print("WARNING: InsightFace is not available (missing insightface or onnxruntime).")
            print("WARNING: Using MOCK embeddings for smoke tests only, not for the final report.")
            self.available = False
        except Exception as exc:
            print(f"WARNING: InsightFace model init failed for {self.model_name}: {exc}")
            print("WARNING: Using MOCK embeddings for smoke tests only, not for the final report.")
            self.available = False

    def get_embedding(self, image):
        if not self.available:
            img_resized = cv2.resize(image, (112, 112)).flatten().astype(np.float32)
            raw_norm = float(np.linalg.norm(img_resized) + 1e-8)
            emb = img_resized / raw_norm
            return emb, raw_norm

        faces = self.app.get(image)
        if not faces:
            return None, 0.0
        face = max(faces, key=lambda x: x.det_score)
        emb = face.embedding
        raw_norm = float(np.linalg.norm(emb))
        emb = emb / raw_norm
        return emb, raw_norm

    def similarity(self, img1, img2):
        emb1, norm1 = self.get_embedding(img1)
        emb2, norm2 = self.get_embedding(img2)
        if emb1 is None or emb2 is None:
            return None, None, None
        sim = float(np.dot(emb1, emb2))
        q = np.clip((norm1 + norm2) / 2 / 25.0, 0.01, 1.0)
        return sim, q, (norm1 + norm2) / 2
