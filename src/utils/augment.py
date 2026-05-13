"""
Augmentation — Synthetic Environmental Degradation
Từ ảnh BRIGHT thật → sinh DARK và MEDIUM theo phương pháp trong báo cáo:
  - Gamma correction (điều chỉnh độ sáng phi tuyến)
  - Gaussian noise (nhiễu cảm biến camera)
  - Motion blur (rung camera)
  - JPEG compression artifacts (camera edge thực tế)
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional


class SyntheticAugmentor:
    # Preset cho từng điều kiện
    PRESETS = {
        'bright': {
            'gamma': (0.85, 1.15),
            'noise_sigma': (0, 5),
            'blur_prob': 0.0,
            'jpeg_quality': (90, 100),
        },
        'medium': {
            'gamma': (0.55, 0.75),
            'noise_sigma': (3, 12),
            'blur_prob': 0.2,
            'jpeg_quality': (70, 90),
        },
        'dark': {
            'gamma': (0.25, 0.45),
            'noise_sigma': (10, 25),
            'blur_prob': 0.4,
            'jpeg_quality': (50, 75),
        },
        'dark_extreme': {
            'gamma': (0.10, 0.25),
            'noise_sigma': (20, 40),
            'blur_prob': 0.6,
            'jpeg_quality': (40, 65),
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma correction: làm tối (gamma > 1) hoặc sáng (gamma < 1)
        Convention báo cáo: gamma nhỏ → ảnh tối (power = 1/gamma > 1)"""
        img = image.astype(np.float32) / 255.0
        img = np.power(img, 1.0 / gamma)  # gamma=0.4 → power=2.5 → tối
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def apply_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian noise mô phỏng ISO noise của camera"""
        noise = self.rng.normal(0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, image: np.ndarray,
                          kernel_size: int = 5,
                          angle: float = None) -> np.ndarray:
        """Motion blur mô phỏng rung camera hoặc người đi qua nhanh"""
        if angle is None:
            angle = float(self.rng.uniform(0, 180))
        k = max(3, kernel_size)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= kernel.sum() + 1e-8
        return cv2.filter2D(image, -1, kernel)

    def apply_jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """JPEG compression artifacts (edge camera thực tế thường nén nhiều)"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def augment(self, image: np.ndarray,
                condition: str = 'dark',
                n_variants: int = 5) -> List[np.ndarray]:
        """
        Sinh n_variants ảnh từ 1 ảnh gốc theo điều kiện chỉ định.

        Args:
            image: ảnh BGR gốc (bright quality)
            condition: 'bright' | 'medium' | 'dark' | 'dark_extreme'
            n_variants: số ảnh sinh ra
        Returns:
            list của n_variants ảnh augmented
        """
        preset = self.PRESETS.get(condition, self.PRESETS['dark'])
        results = []

        for _ in range(n_variants):
            img = image.copy()

            # 1. Gamma correction
            gamma = float(self.rng.uniform(*preset['gamma']))
            img = self.apply_gamma(img, gamma)

            # 2. Gaussian noise
            sigma = float(self.rng.uniform(*preset['noise_sigma']))
            if sigma > 0.5:
                img = self.apply_noise(img, sigma)

            # 3. Motion blur (probabilistic)
            if self.rng.random() < preset['blur_prob']:
                k = int(self.rng.integers(3, 7))
                img = self.apply_motion_blur(img, kernel_size=k)

            # 4. JPEG compression
            quality = int(self.rng.integers(*preset['jpeg_quality']))
            img = self.apply_jpeg_compression(img, quality)

            results.append(img)

        return results

    def build_dataset(self,
                      bright_dir: str,
                      output_dir: str,
                      n_per_image: int = 5) -> dict:
        """
        Từ thư mục ảnh bright → sinh thêm medium và dark.

        Cấu trúc output:
          output_dir/
            bright/   (copy từ bright_dir)
            medium/   (augmented)
            dark/     (augmented)
        """
        bright_dir = Path(bright_dir)
        output_dir = Path(output_dir)
        stats = {'bright': 0, 'medium': 0, 'dark': 0}

        for condition in ['bright', 'medium', 'dark']:
            (output_dir / condition).mkdir(parents=True, exist_ok=True)

        # Đọc tất cả ảnh bright
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        bright_images = []
        for f in bright_dir.iterdir():
            if f.suffix.lower() in img_extensions:
                img = cv2.imread(str(f))
                if img is not None:
                    bright_images.append((f.stem, img))

        print(f"Found {len(bright_images)} bright images in {bright_dir}")

        for stem, img in bright_images:
            # Copy ảnh bright gốc
            out_path = output_dir / 'bright' / f"{stem}.jpg"
            cv2.imwrite(str(out_path), img)
            stats['bright'] += 1

            # Sinh medium variants
            medium_imgs = self.augment(img, 'medium', n_per_image)
            for i, m_img in enumerate(medium_imgs):
                out_path = output_dir / 'medium' / f"{stem}_m{i:02d}.jpg"
                cv2.imwrite(str(out_path), m_img)
                stats['medium'] += 1

            # Sinh dark variants
            dark_imgs = self.augment(img, 'dark', n_per_image)
            for i, d_img in enumerate(dark_imgs):
                out_path = output_dir / 'dark' / f"{stem}_d{i:02d}.jpg"
                cv2.imwrite(str(out_path), d_img)
                stats['dark'] += 1

        print(f"Dataset built: {stats}")
        return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n',      type=int, default=5)
    args = parser.parse_args()

    if args.input and args.output:
        # Chế độ build dataset thật
        aug = SyntheticAugmentor()
        stats = aug.build_dataset(args.input, args.output, n_per_image=args.n)
        print(f"Done: {stats}")
    else:
        # Sanity test
        aug = SyntheticAugmentor()
        test_img = np.zeros((112, 112, 3), dtype=np.uint8)
        for i in range(112):
            test_img[i, :] = int(200 * i / 112)
        variants = aug.augment(test_img, condition='dark', n_variants=3)
        print(f"Generated {len(variants)} dark variants")
        for i, v in enumerate(variants):
            print(f"  Variant {i}: mean_luminance={v.mean()/255.0:.3f}")
        print("Module Augmentor OK")