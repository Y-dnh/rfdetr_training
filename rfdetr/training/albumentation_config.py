"""
Albumentations transform list for RF-DETR training.

Full list of transforms from https://explore.albumentations.ai/
(Image-Only and Dual). Each transform has default parameters and a short comment.
Use this list as ALBUMENTATION_CONFIG; pass it to AlbumentationsWrapper.
A.Mosaic is included (used in pipeline when use_albumentations_mosaic=True);
legacy Mosaic in mosaic.py remains as fallback.
"""

from __future__ import annotations

import inspect
from typing import Any, List

# Lazy import so albumentations is optional until wrapper is used
def _get_albumentations():
    import albumentations as A
    return A


def _safe_append(transforms: List[Any], factory: Any) -> None:
    """Додати трансформ з factory (callable або інстанс). Пропускати тільки при винятку."""
    try:
        t = factory() if callable(factory) else factory
        if t is not None:
            transforms.append(t)
    except Exception:
        pass


def get_default_albu_config(imgsz: int = 640) -> List[Any]:
    """
    Повертає дефолтний список A.* трансформ (для випадку коли в train.py ALBUMENTATION_CONFIG=None).
    imgsz: роздільність (train — з моделі, preview — 640).
    Якщо хочеш свій набір — в train.py пиши просто: ALBUMENTATION_CONFIG = [A.Blur(...), A.GaussNoise(...), ...].
    """
    try:
        A = _get_albumentations()
    except ImportError:
        return []

    crop_sz = max(64, int(imgsz * 0.6))
    transforms: List[Any] = []

    # -------------------------------------------------------------------------
    # Image-Only Transforms
    # -------------------------------------------------------------------------
    if hasattr(A, "AdditiveNoise"):
        def _additive_noise():
            sig = inspect.signature(A.AdditiveNoise).parameters
            if "noise_params" in sig:
                return A.AdditiveNoise(noise_params=(5.0, 15.0), p=0.0)
            return A.AdditiveNoise(scale=(5, 15), p=0.0)
        _safe_append(transforms, _additive_noise)  # Additive noise
    if hasattr(A, "AdvancedBlur"):
        _safe_append(transforms, A.AdvancedBlur(blur_limit=(3, 7), p=0.0))  # Advanced Gaussian blur
    if hasattr(A, "AutoContrast"):
        _safe_append(transforms, A.AutoContrast(p=0.0))  # Auto contrast
    if hasattr(A, "Blur"):
        _safe_append(transforms, A.Blur(blur_limit=7, p=0.2))
    if hasattr(A, "CLAHE"):
        _safe_append(transforms, A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5))
    if hasattr(A, "ChannelDropout"):
        _safe_append(transforms, A.ChannelDropout(channel_drop_range=(1, 1), p=0.0))  # Drop channel(s)
    if hasattr(A, "ChannelShuffle"):
        _safe_append(transforms, A.ChannelShuffle(p=0.0))  # Shuffle RGB channels
    if hasattr(A, "ChromaticAberration"):
        _safe_append(transforms, A.ChromaticAberration(p=0.0))  # Chromatic aberration
    if hasattr(A, "ColorJitter"):
        _safe_append(transforms, A.ColorJitter(p=0.0))  # Color jitter
    if hasattr(A, "Defocus"):
        _safe_append(transforms, A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=0.0))  # Defocus
    if hasattr(A, "Downscale"):
        def _downscale():
            sig = inspect.signature(A.Downscale).parameters
            if "scale_range" in sig:
                return A.Downscale(scale_range=(0.25, 0.5), p=0.3)
            return A.Downscale(scale_min=0.25, scale_max=0.5, p=0.3)
        _safe_append(transforms, _downscale)
    if hasattr(A, "Dithering"):
        _safe_append(transforms, lambda: A.Dithering(p=0.0))  # Dithering (if exists)
    if hasattr(A, "Emboss"):
        _safe_append(transforms, A.Emboss(strength=(0.2, 0.7), p=0.0))  # Emboss
    if hasattr(A, "Equalize"):
        _safe_append(transforms, A.Equalize(p=0.0))  # Histogram equalize
    if hasattr(A, "FDA"):
        _safe_append(transforms, lambda: A.FDA(reference_images=[], beta_limit=(0.0, 0.1), p=0.0))  # Frequency Domain Adaptation
    if hasattr(A, "FancyPCA"):
        _safe_append(transforms, A.FancyPCA(alpha=0.1, p=0.0))  # FancyPCA
    if hasattr(A, "FromFloat"):
        _safe_append(transforms, A.FromFloat(dtype="uint8", max_value=255.0, p=0.0))  # FromFloat (we usually don't need)
    if hasattr(A, "GaussNoise"):
        def _gauss_noise():
            sig = inspect.signature(A.GaussNoise).parameters
            if "std_range" in sig:
                return A.GaussNoise(std_range=(10.0, 50.0), mean_range=(0.0, 0.0), p=0.3)
            return A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
        _safe_append(transforms, _gauss_noise)
    if hasattr(A, "GaussianBlur"):
        _safe_append(transforms, A.GaussianBlur(blur_limit=(3, 7), p=0.1))
    if hasattr(A, "GlassBlur"):
        _safe_append(transforms, A.GlassBlur(sigma=0.7, max_delta=4, p=0.0))  # Glass blur
    if hasattr(A, "HEStain"):
        _safe_append(transforms, lambda: A.HEStain(p=0.0))  # H&E stain augmentation
    if hasattr(A, "HistogramMatching"):
        _safe_append(transforms, lambda: A.HistogramMatching(reference_images=[], blend_ratio=(0.5, 1.0), p=0.0))
    if hasattr(A, "HueSaturationValue"):
        _safe_append(transforms, A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
    if hasattr(A, "ISONoise"):
        _safe_append(transforms, A.ISONoise(intensity=(0.1, 0.5), p=0.0))  # ISO noise
    if hasattr(A, "Illumination"):
        _safe_append(transforms, lambda: A.Illumination(p=0.0))  # Illumination
    if hasattr(A, "ImageCompression"):
        def _image_compression():
            sig = inspect.signature(A.ImageCompression).parameters
            if "quality_range" in sig:
                return A.ImageCompression(quality_range=(75, 100), p=0.0)
            return A.ImageCompression(quality_lower=75, quality_upper=100, p=0.0)
        _safe_append(transforms, _image_compression)  # JPEG compression
    if hasattr(A, "InvertImg"):
        _safe_append(transforms, A.InvertImg(p=0.0))  # Invert image
    if hasattr(A, "MedianBlur"):
        _safe_append(transforms, A.MedianBlur(blur_limit=7, p=0.0))  # Median blur
    if hasattr(A, "MotionBlur"):
        _safe_append(transforms, A.MotionBlur(blur_limit=7, p=0.0))  # Motion blur
    if hasattr(A, "MultiplicativeNoise"):
        _safe_append(transforms, A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.0))  # Multiplicative noise
    # Skip A.Normalize - we do our own in pipeline
    if hasattr(A, "PhotoMetricDistort"):
        _safe_append(transforms, lambda: A.PhotoMetricDistort(p=0.0))  # PhotoMetricDistort (if exists)
    if hasattr(A, "PixelDistributionAdaptation"):
        _safe_append(transforms, lambda: A.PixelDistributionAdaptation(reference_images=[], blend_ratio=(0.5, 1.0), p=0.0))
    if hasattr(A, "PlanckianJitter"):
        _safe_append(transforms, lambda: A.PlanckianJitter(p=0.0))  # PlanckianJitter
    if hasattr(A, "PlasmaBrightnessContrast"):
        _safe_append(transforms, lambda: A.PlasmaBrightnessContrast(p=0.0))  # PlasmaBrightnessContrast
    if hasattr(A, "PlasmaShadow"):
        _safe_append(transforms, lambda: A.PlasmaShadow(p=0.0))  # PlasmaShadow
    if hasattr(A, "Posterize"):
        _safe_append(transforms, A.Posterize(num_bits=4, p=0.0))  # Posterize
    if hasattr(A, "RGBShift"):
        _safe_append(transforms, A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3))
    if hasattr(A, "RandomBrightnessContrast"):
        _safe_append(transforms, A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6))
    if hasattr(A, "RandomFog"):
        def _random_fog():
            sig = inspect.signature(A.RandomFog).parameters
            if "fog_coef_range" in sig:
                return A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=0.25)
            return A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.25)
        _safe_append(transforms, _random_fog)
    if hasattr(A, "RandomGamma"):
        _safe_append(transforms, A.RandomGamma(gamma_limit=(80, 120), p=0.0))  # Random gamma
    if hasattr(A, "RandomGravel"):
        _safe_append(transforms, lambda: A.RandomGravel(number_of_patches=1, p=0.2))
    if hasattr(A, "RandomRain"):
        _safe_append(transforms, A.RandomRain(p=0.0))  # Rain effect
    if hasattr(A, "RandomShadow"):
        _safe_append(transforms, A.RandomShadow(p=0.0))  # Shadow
    if hasattr(A, "RandomSnow"):
        _safe_append(transforms, A.RandomSnow(p=0.0))  # Snow
    if hasattr(A, "RandomSunFlare"):
        _safe_append(transforms, A.RandomSunFlare(p=0.0))  # Sun flare
    if hasattr(A, "RandomToneCurve"):
        _safe_append(transforms, A.RandomToneCurve(scale=0.1, p=0.0))  # RandomToneCurve
    if hasattr(A, "RingingOvershoot"):
        _safe_append(transforms, A.RingingOvershoot(blur_limit=(3, 7), cutoff=(0.5, 1.0), p=0.0))  # RingingOvershoot
    if hasattr(A, "SaltAndPepper"):
        def _salt_pepper():
            sig = inspect.signature(A.SaltAndPepper).parameters
            if "amount" in sig:
                return A.SaltAndPepper(amount=0.02, salt_vs_pepper=0.5, p=0.0)
            return A.SaltAndPepper(noise_density=0.02, p=0.0)
        _safe_append(transforms, _salt_pepper)  # Salt and pepper noise
    if hasattr(A, "Sharpen"):
        _safe_append(transforms, A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.0))  # Sharpen
    if hasattr(A, "ShotNoise"):
        _safe_append(transforms, A.ShotNoise(scale_range=(10, 50), p=0.0))  # ShotNoise
    if hasattr(A, "Solarize"):
        def _solarize():
            sig = inspect.signature(A.Solarize).parameters
            if "threshold_range" in sig:
                return A.Solarize(threshold_range=(128, 255), p=0.0)
            return A.Solarize(threshold=128, p=0.0)
        _safe_append(transforms, _solarize)  # Solarize
    if hasattr(A, "Spatter"):
        _safe_append(transforms, A.Spatter(intensity=(0.1, 0.3), p=0.0))  # Spatter
    if hasattr(A, "Superpixels"):
        _safe_append(transforms, A.Superpixels(p_replace=0.1, n_segments=100, p=0.0))  # Superpixels
    if hasattr(A, "ToFloat"):
        _safe_append(transforms, A.ToFloat(max_value=255.0, p=0.0))  # ToFloat
    if hasattr(A, "ToGray"):
        _safe_append(transforms, A.ToGray(p=0.0))  # Convert to grayscale
    if hasattr(A, "ToRGB"):
        _safe_append(transforms, A.ToRGB(p=0.0))  # Ensure RGB
    if hasattr(A, "ToSepia"):
        _safe_append(transforms, A.ToSepia(p=0.0))  # Sepia
    if hasattr(A, "UnsharpMask"):
        _safe_append(transforms, A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.0, p=0.0))  # Unsharp mask
    if hasattr(A, "ZoomBlur"):
        _safe_append(transforms, A.ZoomBlur(max_factor=1.31, p=0.0))  # Zoom blur

    # -------------------------------------------------------------------------
    # Dual Transforms (affect bboxes; need bbox_params in Compose)
    # -------------------------------------------------------------------------
    if hasattr(A, "Affine"):
        _safe_append(transforms, A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=10, shear=5, p=0.0))  # Affine
    if hasattr(A, "AtLeastOneBBoxRandomCrop"):
        _safe_append(transforms, lambda: A.AtLeastOneBBoxRandomCrop(height=crop_sz, width=crop_sz, p=0.0))  # AtLeastOneBBoxRandomCrop
    if hasattr(A, "BBoxSafeRandomCrop"):
        _safe_append(transforms, lambda: A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.0))  # BBoxSafeRandomCrop
    if hasattr(A, "CenterCrop"):
        _safe_append(transforms, A.CenterCrop(height=crop_sz, width=crop_sz, p=0.0))  # Center crop
    if hasattr(A, "CoarseDropout"):
        def _coarse_dropout():
            sig = inspect.signature(A.CoarseDropout).parameters
            if "num_holes_range" in sig:
                fill = 128 if "fill" in sig else None
                kw = dict(num_holes_range=(4, 12), hole_height_range=(16, 48), hole_width_range=(16, 48), p=0.5)
                if fill is not None:
                    kw["fill"] = 128
                else:
                    kw["fill_value"] = 128
                return A.CoarseDropout(**kw)
            return A.CoarseDropout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=128, p=0.5)
        _safe_append(transforms, _coarse_dropout)
    if hasattr(A, "ConstrainedCoarseDropout"):
        _safe_append(transforms, lambda: A.ConstrainedCoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.0))
    if hasattr(A, "Crop"):
        _safe_append(transforms, lambda: A.Crop(x_min=0, y_min=0, x_max=100, y_max=100, p=0.0))  # Crop (p=0: unused)
    if hasattr(A, "CropAndPad"):
        _safe_append(transforms, A.CropAndPad(percent=0, p=0.0))  # CropAndPad (only percent or px)
    if hasattr(A, "CropNonEmptyMaskIfExists"):
        _safe_append(transforms, lambda: A.CropNonEmptyMaskIfExists(p=0.0))  # CropNonEmptyMaskIfExists
    if hasattr(A, "D4"):
        _safe_append(transforms, A.D4(p=0.0))  # D4 symmetry
    if hasattr(A, "ElasticTransform"):
        _safe_append(transforms, A.ElasticTransform(alpha=1, sigma=50, p=0.0))  # ElasticTransform
    if hasattr(A, "Erasing"):
        _safe_append(transforms, lambda: A.Erasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), fill=128, p=0.25))
    if hasattr(A, "FrequencyMasking"):
        _safe_append(transforms, A.FrequencyMasking(freq_mask_param=10, p=0.0))  # FrequencyMasking (spectrogram)
    if hasattr(A, "GridDistortion"):
        _safe_append(transforms, A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.0))  # GridDistortion
    if hasattr(A, "GridDropout"):
        _safe_append(transforms, lambda: A.GridDropout(ratio=0.1, p=0.3))
    if hasattr(A, "GridElasticDeform"):
        _safe_append(transforms, A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=1, p=0.0))  # GridElasticDeform
    if hasattr(A, "HorizontalFlip"):
        _safe_append(transforms, A.HorizontalFlip(p=0.5))
    if hasattr(A, "VerticalFlip"):
        _safe_append(transforms, A.VerticalFlip(p=0.3))
    if hasattr(A, "LongestMaxSize"):
        _safe_append(transforms, A.LongestMaxSize(max_size=imgsz, p=0.0))  # LongestMaxSize (we use LetterBox)
    if hasattr(A, "MaskDropout"):
        _safe_append(transforms, lambda: A.MaskDropout(max_objects=1, p=0.0))  # MaskDropout
    if hasattr(A, "Morphological"):
        _safe_append(transforms, A.Morphological(scale=(1, 1), operation="dilation", p=0.0))  # Morphological (erosion/dilation)
    if hasattr(A, "Mosaic"):
        # target_size must be <= cell_shape * grid_yx. Legacy Mosaic in mosaic.py is fallback.
        cell = max(64, imgsz // 2)
        _safe_append(transforms, lambda: A.Mosaic(grid_yx=(2, 2), cell_shape=(cell, cell), target_size=(imgsz, imgsz), metadata_key="mosaic_metadata", p=0.0))  # Mosaic (multi-image)
    if hasattr(A, "NoOp"):
        _safe_append(transforms, A.NoOp(p=0.0))  # No operation
    if hasattr(A, "OpticalDistortion"):
        def _optical_distortion():
            sig = inspect.signature(A.OpticalDistortion).parameters
            if "shift_limit" in sig:
                return A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2)
            return A.OpticalDistortion(distort_limit=0.05, p=0.2)
        _safe_append(transforms, _optical_distortion)
    if hasattr(A, "Pad"):
        _safe_append(transforms, A.Pad(padding=0, fill=0, p=0.0))  # Pad
    if hasattr(A, "PadIfNeeded"):
        _safe_append(transforms, A.PadIfNeeded(min_height=imgsz, min_width=imgsz, p=0.0))  # Pad if needed
    if hasattr(A, "Perspective"):
        _safe_append(transforms, A.Perspective(scale=(0.05, 0.1), p=0.3))
    if hasattr(A, "PixelDropout"):
        _safe_append(transforms, A.PixelDropout(dropout_prob=0.01, p=0.0))  # Pixel dropout
    if hasattr(A, "RandomCrop"):
        _safe_append(transforms, A.RandomCrop(height=crop_sz, width=crop_sz, p=0.0))  # RandomCrop
    if hasattr(A, "RandomCropFromBorders"):
        _safe_append(transforms, A.RandomCropFromBorders(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, p=0.0))
    if hasattr(A, "RandomCropNearBBox"):
        _safe_append(transforms, lambda: A.RandomCropNearBBox(p=0.0))  # RandomCropNearBBox (needs bbox)
    if hasattr(A, "RandomGridShuffle"):
        _safe_append(transforms, A.RandomGridShuffle(grid=(3, 3), p=0.0))  # RandomGridShuffle
    if hasattr(A, "RandomResizedCrop"):
        _safe_append(transforms, A.RandomResizedCrop(size=(imgsz, imgsz), scale=(0.8, 1.0), p=0.0))  # RandomResizedCrop
    if hasattr(A, "RandomRotate90"):
        _safe_append(transforms, A.RandomRotate90(p=0.25))
    if hasattr(A, "RandomScale"):
        _safe_append(transforms, A.RandomScale(scale_limit=(0.9, 1.1), p=0.0))  # RandomScale
    if hasattr(A, "RandomSizedBBoxSafeCrop"):
        _safe_append(transforms, lambda: A.RandomSizedBBoxSafeCrop(height=imgsz, width=imgsz, p=0.0))  # RandomSizedBBoxSafeCrop
    if hasattr(A, "RandomSizedCrop"):
        _safe_append(transforms, lambda: A.RandomSizedCrop(min_max_height=(crop_sz, imgsz), size=imgsz, w2h_ratio=1.0, p=0.0))  # RandomSizedCrop
    if hasattr(A, "Resize"):
        _safe_append(transforms, A.Resize(height=imgsz, width=imgsz, p=0.0))  # Resize (we use LetterBox instead)
    if hasattr(A, "Rotate"):
        _safe_append(transforms, A.Rotate(limit=15, p=0.0))  # Rotate
    if hasattr(A, "SafeRotate"):
        _safe_append(transforms, A.SafeRotate(limit=15, p=0.0))  # Safe rotate (keeps full image)
    if hasattr(A, "ShiftScaleRotate"):
        _safe_append(transforms, A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=10, p=0.5))
    if hasattr(A, "SmallestMaxSize"):
        _safe_append(transforms, A.SmallestMaxSize(max_size=imgsz, p=0.0))  # SmallestMaxSize
    if hasattr(A, "SquareSymmetry"):
        _safe_append(transforms, A.SquareSymmetry(p=0.0))  # SquareSymmetry
    if hasattr(A, "ThinPlateSpline"):
        _safe_append(transforms, A.ThinPlateSpline(scale_range=(0.0, 0.0), num_control_points=4, p=0.0))  # ThinPlateSpline
    if hasattr(A, "TimeMasking"):
        _safe_append(transforms, A.TimeMasking(time_mask_param=10, p=0.0))  # TimeMasking (spectrogram)
    if hasattr(A, "TimeReverse"):
        _safe_append(transforms, A.TimeReverse(p=0.0))  # TimeReverse (spectrogram)
    if hasattr(A, "Transpose"):
        _safe_append(transforms, A.Transpose(p=0.0))  # Transpose (H<->W)
    if hasattr(A, "XYMasking"):
        _safe_append(transforms, A.XYMasking(num_masks_x=1, num_masks_y=1, mask_x_length=10, mask_y_length=10, p=0.0))  # XYMasking (spectrogram)

    return transforms


# Default ALBUMENTATION_CONFIG: call get_default_albu_config() to get the list,
# or assign a custom list of A.* instances in train.py.
# Example in train.py:
#   from rfdetr.training.albumentation_config import get_default_albu_config
#   ALBUMENTATION_CONFIG = get_default_albu_config()
# Or override with custom list:
#   import albumentations as A
#   ALBUMENTATION_CONFIG = [
#       A.Blur(blur_limit=7, p=0.5),
#       A.CLAHE(clip_limit=4.0, p=0.5),
#       ...
#   ]

__all__ = ["get_default_albu_config"]
