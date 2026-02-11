"""
Color augmentations for RF-DETR training.

This module provides color-space augmentations:
- RandomHSV: Hue, Saturation, Value adjustments (Ultralytics-style)
"""

import random
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch

from rfdetr.training.augmentations.base import BaseTransform


class RandomHSV(BaseTransform):
    """
    Random HSV (Hue, Saturation, Value) augmentation.
    
    This augmentation randomly adjusts the Hue, Saturation, and Value
    channels of the image. Implementation follows Ultralytics YOLO style.
    
    Note: This augmentation does NOT modify bounding boxes as it only
    affects pixel colors, not spatial positions.
    
    Args:
        h_gain: Hue gain factor in range [0.0, 1.0]. Hue is shifted by a random
                value in [-h_gain, h_gain] * 180 degrees.
        s_gain: Saturation gain factor in range [0.0, 1.0]. Saturation is scaled
                by a random value in [1-s_gain, 1+s_gain].
        v_gain: Value gain factor in range [0.0, 1.0]. Value is scaled by a
                random value in [1-v_gain, 1+v_gain].
        p: Probability of applying the augmentation.
    
    Example:
        >>> transform = RandomHSV(h_gain=0.015, s_gain=0.7, v_gain=0.4, p=1.0)
        >>> image, target = transform(image, target)
    """
    
    def __init__(
        self, 
        h_gain: float = 0.015, 
        s_gain: float = 0.7, 
        v_gain: float = 0.4,
        p: float = 1.0
    ):
        super().__init__(p=p, name='RandomHSV')
        
        # Validate gains
        if not 0.0 <= h_gain <= 1.0:
            raise ValueError(f"h_gain must be in [0.0, 1.0], got {h_gain}")
        if not 0.0 <= s_gain <= 1.0:
            raise ValueError(f"s_gain must be in [0.0, 1.0], got {s_gain}")
        if not 0.0 <= v_gain <= 1.0:
            raise ValueError(f"v_gain must be in [0.0, 1.0], got {v_gain}")
        
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        
        # Store last applied values for logging
        self._last_h_shift = 0.0
        self._last_s_scale = 1.0
        self._last_v_scale = 1.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """
        Apply HSV augmentation to the image.
        
        Args:
            image: Input image (PIL or numpy RGB format).
            target: Target dict (unchanged as HSV doesn't affect boxes).
        
        Returns:
            Tuple of (augmented_image, target).
        """
        # Convert to numpy if PIL
        is_pil = isinstance(image, PIL.Image.Image)
        if is_pil:
            img = np.array(image)
        else:
            img = image.copy() if isinstance(image, np.ndarray) else image
        
        # Skip if any gain is zero
        if self.h_gain == 0 and self.s_gain == 0 and self.v_gain == 0:
            return image, target
        
        # Generate random gains
        # r is in range [-1, 1] for each channel
        r = np.random.uniform(-1, 1, 3) * np.array([self.h_gain, self.s_gain, self.v_gain]) + 1
        
        # Store for logging
        self._last_h_shift = (r[0] - 1) * 180  # Convert to degrees
        self._last_s_scale = r[1]
        self._last_v_scale = r[2]
        
        # Convert RGB to HSV
        # Note: OpenCV uses BGR by default, but we're working with RGB
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Split channels
        h, s, v = cv2.split(img_hsv)
        
        # Apply gains using LUT (Look-Up Table) for speed
        # Hue: circular range [0, 180) in OpenCV
        dtype = img_hsv.dtype
        
        # Create LUTs
        lut_h = np.arange(0, 256, dtype=np.int16)
        lut_h = ((lut_h * r[0]) % 180).astype(dtype)
        
        lut_s = np.arange(0, 256, dtype=np.int16)
        lut_s = np.clip(lut_s * r[1], 0, 255).astype(dtype)
        
        lut_v = np.arange(0, 256, dtype=np.int16)
        lut_v = np.clip(lut_v * r[2], 0, 255).astype(dtype)
        
        # Apply LUTs
        h = cv2.LUT(h, lut_h)
        s = cv2.LUT(s, lut_s)
        v = cv2.LUT(v, lut_v)
        
        # Merge channels
        img_hsv = cv2.merge([h, s, v])
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to PIL if input was PIL
        if is_pil:
            img_rgb = PIL.Image.fromarray(img_rgb)
        
        return img_rgb, target
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used for the last augmentation."""
        return {
            'h_gain': self.h_gain,
            's_gain': self.s_gain,
            'v_gain': self.v_gain,
            'h_shift_degrees': self._last_h_shift,
            's_scale': self._last_s_scale,
            'v_scale': self._last_v_scale,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"h_gain={self.h_gain}, s_gain={self.s_gain}, v_gain={self.v_gain}, p={self.p})"
        )


class RandomBrightness(BaseTransform):
    """
    Random brightness adjustment.
    
    Args:
        brightness_range: Range of brightness multiplier (min, max).
                         Values < 1 darken, > 1 brighten.
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self, 
        brightness_range: Tuple[float, float] = (0.5, 1.5),
        p: float = 0.5
    ):
        super().__init__(p=p, name='RandomBrightness')
        self.brightness_range = brightness_range
        self._last_factor = 1.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply brightness adjustment."""
        is_pil = isinstance(image, PIL.Image.Image)
        if is_pil:
            img = np.array(image).astype(np.float32)
        else:
            img = image.astype(np.float32)
        
        # Random brightness factor
        factor = random.uniform(*self.brightness_range)
        self._last_factor = factor
        
        img = img * factor
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        if is_pil:
            img = PIL.Image.fromarray(img)
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'brightness_range': self.brightness_range, 'factor': self._last_factor}


class RandomContrast(BaseTransform):
    """
    Random contrast adjustment.
    
    Args:
        contrast_range: Range of contrast multiplier (min, max).
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self, 
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        p: float = 0.5
    ):
        super().__init__(p=p, name='RandomContrast')
        self.contrast_range = contrast_range
        self._last_factor = 1.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply contrast adjustment."""
        is_pil = isinstance(image, PIL.Image.Image)
        if is_pil:
            img = np.array(image).astype(np.float32)
        else:
            img = image.astype(np.float32)
        
        # Random contrast factor
        factor = random.uniform(*self.contrast_range)
        self._last_factor = factor
        
        # Adjust contrast around mean
        mean = img.mean()
        img = (img - mean) * factor + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        if is_pil:
            img = PIL.Image.fromarray(img)
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'contrast_range': self.contrast_range, 'factor': self._last_factor}


class RandomBlur(BaseTransform):
    """
    Random Gaussian blur.
    
    Args:
        kernel_size_range: Range of kernel sizes (min, max). Must be odd numbers.
        p: Probability of applying the augmentation.
    """
    
    def __init__(
        self, 
        kernel_size_range: Tuple[int, int] = (3, 7),
        p: float = 0.1
    ):
        super().__init__(p=p, name='RandomBlur')
        self.kernel_size_range = kernel_size_range
        self._last_kernel_size = 0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply Gaussian blur."""
        is_pil = isinstance(image, PIL.Image.Image)
        if is_pil:
            img = np.array(image)
        else:
            img = image.copy()
        
        # Random kernel size (must be odd)
        k = random.randint(self.kernel_size_range[0] // 2, self.kernel_size_range[1] // 2) * 2 + 1
        self._last_kernel_size = k
        
        img = cv2.GaussianBlur(img, (k, k), 0)
        
        if is_pil:
            img = PIL.Image.fromarray(img)
        
        return img, target
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'kernel_size_range': self.kernel_size_range, 'kernel_size': self._last_kernel_size}


class RandomNoise(BaseTransform):
    """
    Random noise augmentation with multiple noise types.
    
    Supported noise types:
        - 'gaussian_mono': Monochrome Gaussian noise (same noise per channel).
          Realistic for IR/thermal cameras and grayscale images.
        - 'gaussian_rgb': Per-channel Gaussian noise (independent noise per channel).
          Realistic for color RGB cameras.
        - 'salt_pepper': Salt-and-pepper (impulse) noise.
          Random pixels become black (0) or white (255).
          Characteristic for sensor defects and IR cameras.
    
    Args:
        noise_type: Type of noise ('gaussian_mono', 'gaussian_rgb', 'salt_pepper').
        noise_range: Range of noise standard deviation (min, max) for Gaussian types.
        salt_pepper_amount: Fraction of pixels affected (0-1) for salt_pepper type.
        p: Probability of applying the augmentation.
    """
    
    VALID_TYPES = ('gaussian_mono', 'gaussian_rgb', 'salt_pepper')
    
    def __init__(
        self, 
        noise_type: str = 'gaussian_mono',
        noise_range: Tuple[float, float] = (5, 30),
        salt_pepper_amount: float = 0.02,
        p: float = 0.1,
    ):
        super().__init__(p=p, name='RandomNoise')
        
        if noise_type not in self.VALID_TYPES:
            raise ValueError(
                f"noise_type must be one of {self.VALID_TYPES}, got '{noise_type}'"
            )
        
        self.noise_type = noise_type
        self.noise_range = noise_range
        self.salt_pepper_amount = salt_pepper_amount
        self._last_std = 0.0
        self._last_amount = 0.0
    
    def apply(
        self, 
        image: Union[PIL.Image.Image, np.ndarray], 
        target: Dict[str, Any]
    ) -> Tuple[Union[PIL.Image.Image, np.ndarray], Dict[str, Any]]:
        """Apply noise augmentation."""
        is_pil = isinstance(image, PIL.Image.Image)
        if is_pil:
            img = np.array(image)
        else:
            img = image.copy()
        
        if self.noise_type == 'gaussian_mono':
            img = self._apply_gaussian_mono(img)
        elif self.noise_type == 'gaussian_rgb':
            img = self._apply_gaussian_rgb(img)
        elif self.noise_type == 'salt_pepper':
            img = self._apply_salt_pepper(img)
        
        if is_pil:
            img = PIL.Image.fromarray(img)
        
        return img, target
    
    def _apply_gaussian_mono(self, img: np.ndarray) -> np.ndarray:
        """Monochrome Gaussian noise -- same noise value for all channels per pixel."""
        h, w = img.shape[:2]
        std = random.uniform(*self.noise_range)
        self._last_std = std
        
        # Generate single-channel noise and broadcast to all channels
        noise = np.random.normal(0, std, (h, w)).astype(np.float32)
        if img.ndim == 3:
            noise = noise[:, :, np.newaxis]
        
        result = img.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_gaussian_rgb(self, img: np.ndarray) -> np.ndarray:
        """Per-channel Gaussian noise -- independent noise per channel."""
        std = random.uniform(*self.noise_range)
        self._last_std = std
        
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        result = img.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_salt_pepper(self, img: np.ndarray) -> np.ndarray:
        """Salt-and-pepper noise -- random pixels become black or white."""
        amount = random.uniform(self.salt_pepper_amount * 0.5, self.salt_pepper_amount)
        self._last_amount = amount
        
        result = img.copy()
        h, w = img.shape[:2]
        num_pixels = int(amount * h * w)
        
        # Salt (white pixels)
        salt_y = np.random.randint(0, h, num_pixels)
        salt_x = np.random.randint(0, w, num_pixels)
        result[salt_y, salt_x] = 255
        
        # Pepper (black pixels)
        pepper_y = np.random.randint(0, h, num_pixels)
        pepper_x = np.random.randint(0, w, num_pixels)
        result[pepper_y, pepper_x] = 0
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'noise_type': self.noise_type,
            'noise_range': self.noise_range,
            'std': self._last_std,
            'salt_pepper_amount': self._last_amount,
        }
