# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class Union4M_ResNet45(EncoderDecoderRecognizer):
    """ResNet45 baseline in Union4M"""


@MODELS.register_module()
class Union4M_SwinT(EncoderDecoderRecognizer):
    """Swin Transformer baseline in Union4M"""


@MODELS.register_module()
class Union4M_ViT(EncoderDecoderRecognizer):
    """Vision Transformer baseline in Union4M"""
