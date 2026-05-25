from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class ImagePreprocessor:
    """Optional preprocessing for low-quality scanned images."""

    def __init__(
        self,
        target_dpi: int = 300,
        contrast_factor: float = 1.5,
        sharpen: bool = True,
    ) -> None:
        self.target_dpi = target_dpi
        self.contrast_factor = contrast_factor
        self.sharpen = sharpen

    def enhance(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image = self._adjust_contrast(image)
        if self.sharpen:
            image = self._sharpen(image)
        return image

    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.contrast_factor)

    def _sharpen(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.SHARPEN)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        return np.array(image)
