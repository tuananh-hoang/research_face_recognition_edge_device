"""
Module D — Gallery Manager với Continual Learning
Theo báo cáo Section 3.4:

Cấu trúc:
  gallery[partition][person_id] = list of embeddings
  partitions: 'anchor' (bất biến) + 'bright' + 'medium' + 'dark'

Weighted update:
  w = λ · q · det_score · (1 - max_sim_existing)

Eviction: LRU per partition (trừ anchor)

Metrics: CP-BWT (Condition-Partitioned Backward Transfer)
  CP-BWT = Acc_bright(sau adaptation) - Acc_bright(trước adaptation)
"""

import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class GalleryManager:
    PARTITIONS = ['anchor', 'bright', 'medium', 'dark']

    def __init__(self,
                 k_per_person: int = 20,      # max embeddings per person per partition
                 lambda_lr: float = 0.2,       # learning rate cho weighted update
                 min_update_weight: float = 0.05,  # threshold để thực hiện update
                 anchor_immutable: bool = True):    # anchor không bị evict
        self.k_per_person = k_per_person
        self.lambda_lr = lambda_lr
        self.min_update_weight = min_update_weight
        self.anchor_immutable = anchor_immutable

        # gallery[partition][person_id] = OrderedDict{idx: (emb, timestamp, access_count)}
        self.gallery: Dict[str, Dict[str, OrderedDict]] = {
            p: {} for p in self.PARTITIONS
        }
        self._entry_counter = 0  # global counter để track insertion order

        # Stats
        self.stats = {
            'total_queries': 0,
            'total_updates': 0,
            'total_evictions': 0,
            'rejected_low_weight': 0,
        }

    # ─── Enrollment ────────────────────────────────────────────

    def enroll(self, person_id: str, emb: np.ndarray,
               partition: str = 'anchor') -> bool:
        """
        Thêm embedding vào gallery (enrollment chính thức).
        Anchor: không bị evict, dùng cho ảnh sáng chất lượng cao.
        """
        if partition not in self.PARTITIONS:
            partition = 'anchor'

        emb = self._normalize(emb)

        if person_id not in self.gallery[partition]:
            self.gallery[partition][person_id] = OrderedDict()

        person_gallery = self.gallery[partition][person_id]
        entry_id = self._entry_counter
        self._entry_counter += 1
        person_gallery[entry_id] = {
            'emb': emb,
            'timestamp': time.time(),
            'access_count': 0,
            'last_access': time.time(),
        }

        # Eviction nếu vượt K (chỉ cho non-anchor)
        if not (self.anchor_immutable and partition == 'anchor'):
            self._evict_lru(person_id, partition)

        return True

    # ─── Search ────────────────────────────────────────────────

    def search(self, query_emb: np.ndarray,
               bin_id: str,
               top_k: int = 1) -> Tuple[Optional[str], float]:
        """
        1:N search trong anchor + query_partition.
        Trả về (best_person_id, best_cosine_sim).
        """
        self.stats['total_queries'] += 1
        query_emb = self._normalize(query_emb)

        best_id = None
        best_sim = -1.0

        # Search anchor + bin partition
        search_partitions = ['anchor']
        if bin_id in self.PARTITIONS and bin_id != 'anchor':
            search_partitions.append(bin_id)

        for partition in search_partitions:
            for person_id, entries in self.gallery[partition].items():
                for entry_id, entry_data in entries.items():
                    sim = float(np.dot(query_emb, entry_data['emb']))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = person_id

        # Update access stats cho best match
        if best_id and best_sim > 0.2:
            self._update_access(best_id, best_sim)

        return best_id, best_sim

    # ─── Online Update ─────────────────────────────────────────

    def update(self,
               person_id: str,
               emb: np.ndarray,
               bin_id: str,
               q: float,
               det_score: float,
               sim_to_gallery: float,
               tau: float,
               margin: float = 0.03) -> bool:
        """
        Cập nhật gallery theo weighted update từ báo cáo.

        Trigger conditions:
          - sim ≥ τ + margin (confident accept, không borderline)
          - det_score > 0.7
          - q > 0.15 (ảnh không quá tệ)

        Weight:
          w = λ · q · det_score · (1 - max_sim_existing)
          (max_sim_existing = diversity penalty)
        """
        # Kiểm tra trigger conditions
        if sim_to_gallery < tau + margin:
            return False
        if det_score < 0.7:
            return False
        if q < 0.15:
            return False
        if bin_id == 'anchor':  # không tự update anchor
            return False

        emb = self._normalize(emb)

        # Tính diversity penalty: max cosine sim với embeddings hiện có của người này
        max_sim_existing = self._max_sim_to_existing(person_id, emb)

        # Tính weight
        w = self.lambda_lr * q * det_score * (1.0 - max_sim_existing)

        if w < self.min_update_weight:
            self.stats['rejected_low_weight'] += 1
            return False

        # Thực hiện update
        self.enroll(person_id, emb, partition=bin_id)
        self.stats['total_updates'] += 1
        return True

    # ─── Memory Management ─────────────────────────────────────

    def _evict_lru(self, person_id: str, partition: str) -> None:
        """LRU eviction: loại entry ít được truy cập nhất / cũ nhất"""
        person_gallery = self.gallery[partition].get(person_id, OrderedDict())
        if len(person_gallery) <= self.k_per_person:
            return

        # Sắp xếp theo last_access (cũ nhất trước)
        sorted_entries = sorted(
            person_gallery.items(),
            key=lambda x: x[1]['last_access']
        )

        # Xóa entries cũ nhất cho đến khi còn K
        while len(person_gallery) > self.k_per_person:
            oldest_id = sorted_entries.pop(0)[0]
            del person_gallery[oldest_id]
            self.stats['total_evictions'] += 1

    def _update_access(self, person_id: str, sim: float) -> None:
        """Cập nhật access timestamp cho LRU tracking"""
        now = time.time()
        for partition in self.PARTITIONS:
            if person_id in self.gallery[partition]:
                for entry_data in self.gallery[partition][person_id].values():
                    entry_data['access_count'] += 1
                    entry_data['last_access'] = now
                break

    def _max_sim_to_existing(self, person_id: str, query_emb: np.ndarray) -> float:
        """Tính max cosine sim của query_emb với tất cả embeddings đã có của person_id"""
        max_sim = 0.0
        for partition in self.PARTITIONS:
            if person_id in self.gallery[partition]:
                for entry_data in self.gallery[partition][person_id].values():
                    sim = float(np.dot(query_emb, entry_data['emb']))
                    max_sim = max(max_sim, sim)
        return max_sim

    # ─── Metrics ───────────────────────────────────────────────

    def compute_cp_bwt(self,
                       acc_before: Dict[str, float],
                       acc_after: Dict[str, float]) -> Dict:
        """
        CP-BWT: Condition-Partitioned Backward Transfer
        Metric mới từ báo cáo, đo stability theo condition.

        Args:
            acc_before: {'bright': 0.95, 'medium': 0.88, 'dark': 0.62}
            acc_after:  {'bright': 0.94, 'medium': 0.89, 'dark': 0.71}
        Returns:
            cp_bwt per condition và overall BWT
        """
        cp_bwt = {}
        for condition in ['bright', 'medium', 'dark']:
            before = acc_before.get(condition, 0.0)
            after = acc_after.get(condition, 0.0)
            cp_bwt[condition] = after - before  # dương = improved, âm = forgot

        # Overall BWT (average)
        cp_bwt['overall_bwt'] = np.mean([
            cp_bwt.get(c, 0) for c in ['bright', 'medium', 'dark']
        ])

        # Stability check: bright condition should not degrade
        cp_bwt['stable'] = cp_bwt.get('bright', -1) >= -0.01

        return cp_bwt

    # ─── Utils ─────────────────────────────────────────────────

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def get_size_kb(self) -> float:
        """Tính tổng memory của gallery (KB)"""
        total_vectors = sum(
            len(entries)
            for partition in self.gallery.values()
            for entries in partition.values()
        )
        # Mỗi vector: 512 dims × 4 bytes (float32)
        return total_vectors * 512 * 4 / 1024

    def get_person_count(self) -> Dict[str, int]:
        return {p: len(persons) for p, persons in self.gallery.items()}

    def summary(self) -> None:
        print("\n═══ Gallery Summary ═══════════════════════")
        for partition, persons in self.gallery.items():
            total_entries = sum(len(e) for e in persons.values())
            print(f"  {partition:<8}: {len(persons):>3} persons, "
                  f"{total_entries:>4} entries")
        print(f"  Total memory: {self.get_size_kb():.1f} KB")
        print(f"  Queries: {self.stats['total_queries']}, "
              f"Updates: {self.stats['total_updates']}, "
              f"Evictions: {self.stats['total_evictions']}")
        print("════════════════════════════════════════════\n")


# ─── Sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    gallery = GalleryManager(k_per_person=5)

    # Enroll 3 người với ảnh sáng
    np.random.seed(42)
    for pid in ['alice', 'bob', 'carol']:
        for _ in range(3):
            emb = np.random.randn(512).astype(np.float32)
            gallery.enroll(pid, emb, partition='anchor')

    # Test search
    test_emb = gallery.gallery['anchor']['alice'][0]['emb'].copy()
    noise = np.random.randn(512) * 0.1
    test_emb = test_emb + noise  # noisy version của alice

    best_id, best_sim = gallery.search(test_emb, bin_id='dark')
    print(f"Search result: {best_id} (sim={best_sim:.3f}) — expected: alice")

    # Test online update
    dark_emb = np.random.randn(512).astype(np.float32)
    updated = gallery.update(
        person_id='alice',
        emb=dark_emb,
        bin_id='dark',
        q=0.5,
        det_score=0.85,
        sim_to_gallery=0.52,
        tau=0.44,
        margin=0.03
    )
    print(f"Update triggered: {updated}")

    # CP-BWT test
    acc_before = {'bright': 0.95, 'medium': 0.88, 'dark': 0.62}
    acc_after  = {'bright': 0.94, 'medium': 0.89, 'dark': 0.71}
    bwt = gallery.compute_cp_bwt(acc_before, acc_after)
    print(f"CP-BWT: bright={bwt['bright']:+.3f}, dark={bwt['dark']:+.3f}, "
          f"overall={bwt['overall_bwt']:+.3f}, stable={bwt['stable']}")

    gallery.summary()
    print("Module D OK")