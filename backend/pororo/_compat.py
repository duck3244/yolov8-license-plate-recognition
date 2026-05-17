"""
Vendored pororo가 외부 라이브러리(torchvision, torch, Pillow)의 버전 차이로
깨지지 않도록 한 곳에서 흡수하는 호환 레이어.

새로운 drift가 발생하면 이 파일만 수정하면 됨.
"""
from __future__ import annotations

from typing import Any

import torch
import torchvision.models as tvmodels
from PIL import Image

# torchvision <0.13 의 model_urls는 0.13+에서 제거됨
try:
    from torchvision.models.vgg import model_urls  # noqa: F401
except ImportError:
    model_urls = {"vgg16_bn": ""}


def load_vgg16_bn(pretrained: bool = True):
    """
    torchvision 0.13+ 의 weights API 와 구버전의 pretrained 인자를 모두 지원.
    """
    try:
        weights = tvmodels.VGG16_BN_Weights.DEFAULT if pretrained else None
        return tvmodels.vgg16_bn(weights=weights)
    except (AttributeError, TypeError):
        return tvmodels.vgg16_bn(pretrained=pretrained)


def safe_torch_load(path: str, map_location: Any = None):
    """
    torch.load 의 weights_only 기본값 변경(PyTorch 2.6+)에 대비.

    vendored 모델 가중치는 신뢰 가능하므로 weights_only=False 명시.
    PyTorch 2.6 정식 도입 후 add_safe_globals 기반 안전 모드 재검토.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_pil_antialias() -> None:
    """
    Pillow 10에서 제거된 Image.ANTIALIAS 상수 별칭을 idempotent 하게 부여.
    pororo 내부에서 Image.ANTIALIAS 를 사용하는 모듈이 import 되기 전에 호출.
    """
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]


# import 시점에 자동 적용 (Image.ANTIALIAS 는 read-only 상수가 아님)
ensure_pil_antialias()
