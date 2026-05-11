import numpy as np
import cv2

class RealEmbedder:
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_sc',  # model nhỏ, CPU-friendly
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.available = True
        except ImportError:
            print("⚠️ InsightFace không khả dụng (thiếu module insightface hoặc onnxruntime).")
            print("⚠️ Đang dùng MOCK embedding cho mục đích test (không dùng cho báo cáo NCKH thật).")
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