from __future__ import annotations
import os
from typing import List, Tuple

import cv2
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

ENCODER_FILENAME = "edge_sam_3x_encoder.onnx"
DECODER_FILENAME = "edge_sam_3x_decoder.onnx"
ENCODER_PATH = os.path.join(WEIGHTS_DIR, ENCODER_FILENAME)
DECODER_PATH = os.path.join(WEIGHTS_DIR, DECODER_FILENAME)

SAM2_ENCODER_FILENAME = "sam2_hiera_tiny.encoder.onnx"
SAM2_DECODER_FILENAME = "sam2_hiera_tiny.decoder.onnx"
SAM2_ENCODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_ENCODER_FILENAME)
SAM2_DECODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_DECODER_FILENAME)

SAM2_BASE_PLUS_ENCODER_FILENAME = "sam2_hiera_base_plus.encoder.onnx"
SAM2_BASE_PLUS_DECODER_FILENAME = "sam2_hiera_base_plus.decoder.onnx"
SAM2_BASE_PLUS_ENCODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_BASE_PLUS_ENCODER_FILENAME)
SAM2_BASE_PLUS_DECODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_BASE_PLUS_DECODER_FILENAME)

SAM2_LARGE_ENCODER_FILENAME = "sam2_hiera_large.encoder.onnx"
SAM2_LARGE_DECODER_FILENAME = "sam2_hiera_large.decoder.onnx"
SAM2_LARGE_ENCODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_LARGE_ENCODER_FILENAME)
SAM2_LARGE_DECODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_LARGE_DECODER_FILENAME)

# SAM 2.1 keeps SAM 2's architecture and ONNX signature — same encoder/decoder
# tensor names, shapes and dtypes — so SAM2Predictor drives it unchanged.
SAM2_1_LARGE_ENCODER_FILENAME = "sam2.1_hiera_large.encoder.onnx"
SAM2_1_LARGE_DECODER_FILENAME = "sam2.1_hiera_large.decoder.onnx"
SAM2_1_LARGE_ENCODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_1_LARGE_ENCODER_FILENAME)
SAM2_1_LARGE_DECODER_PATH = os.path.join(WEIGHTS_DIR, SAM2_1_LARGE_DECODER_FILENAME)

_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
_PIXEL_STD  = np.array([58.395,  57.12,  57.375],  dtype=np.float32)[None, :, None, None]
_IMG_SIZE   = 1024

_SAM2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_SAM2_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def is_installed() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _make_session(path: str, providers: List[str]):
    """Build an InferenceSession with ONNX Runtime's warning chatter silenced.

    These exported models produce warnings we cannot act on:
      - at load, from graph.cc — shape-merge fallbacks ("Error merging shape
        info ... Falling back to lenient merge") and unused-initializer notices,
        both artifacts of how the models were exported to ONNX;
      - at inference on CUDA, from scatter_nd.h — a generic caveat that
        ScatterND is only exact when indices are not duplicated.

    Severity 3 is Error, so real failures are still reported; only Warning and
    below are dropped. Two knobs are needed because the load-time messages go to
    the session logger while the CUDA kernel ones go to the process-wide default
    logger, which session options do not reach.
    """
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


class _ResizeLongestSide:
    def __init__(self, target: int = _IMG_SIZE) -> None:
        self._target = target

    def new_hw(self, h: int, w: int) -> Tuple[int, int]:
        scale = self._target / max(h, w)
        return int(h * scale + 0.5), int(w * scale + 0.5)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        nh, nw = self.new_hw(*image.shape[:2])
        return cv2.resize(image, (nw, nh))

    def apply_coords(self, coords: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
        oh, ow = orig_hw
        nh, nw = self.new_hw(oh, ow)
        out = coords.copy().astype(np.float32)
        out[..., 0] *= nw / ow
        out[..., 1] *= nh / oh
        return out


class EdgeSAMPredictor:
    """ONNX Runtime based EdgeSAM predictor — no torch/mmdet/mmcv required."""

    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        import onnxruntime as ort

        # Make cuDNN discoverable for onnxruntime by adding PyTorch's lib dir to PATH.
        # PyTorch bundles cudnn64_9.dll which onnxruntime-gpu 1.20+ needs.
        try:
            import torch, os
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._enc = _make_session(encoder_path, providers)
        self._dec = _make_session(decoder_path, providers)
        self._tf  = _ResizeLongestSide()
        self._features:   np.ndarray | None = None
        self._input_size: Tuple[int, int]   = (0, 0)
        self._orig_size:  Tuple[int, int]   = (0, 0)

    @property
    def device(self) -> str:
        p = self._enc.get_providers()
        return "CUDA" if "CUDAExecutionProvider" in p else "CPU"

    def set_image(self, image_rgb: np.ndarray) -> None:
        h, w = image_rgb.shape[:2]
        self._orig_size = (h, w)

        resized = self._tf.apply_image(image_rgb)
        self._input_size = resized.shape[:2]   # (nh, nw)

        # NCHW float32, normalise, pad to IMG_SIZE × IMG_SIZE
        x = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        x = (x - _PIXEL_MEAN) / _PIXEL_STD
        pad_h = _IMG_SIZE - resized.shape[0]
        pad_w = _IMG_SIZE - resized.shape[1]
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

        self._features = self._enc.run(None, {"image": x})[0]

    def predict(
        self,
        points: List[Tuple[int, int]],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (masks [3,H,W] bool, scores [3] float32).

        The ONNX decoder outputs 4 candidates; index 0 is the single-mask
        best pick, indices 1-3 are the small/medium/large multi-mask set.
        We return only indices 1-3 to match the 3-option UI slider.
        """
        coords = self._tf.apply_coords(
            np.array(points, dtype=np.float32), self._orig_size
        )
        # Decoder expects batch dim: [1, N, 2] and [1, N]
        coords = coords[np.newaxis]
        lbls = np.array(labels, dtype=np.float32)[np.newaxis]

        out = self._dec.run(None, {
            "image_embeddings": self._features,
            "point_coords":     coords,
            "point_labels":     lbls,
        })
        scores  = out[0][0]   # (4,) — remove batch dim
        low_res = out[1][0]   # (4, 256, 256) — remove batch dim

        # Use multi-mask candidates (skip index 0 = single-mask output)
        scores  = scores[1:]    # (3,)
        low_res = low_res[1:]   # (3, 256, 256)

        # Upsample: 256×256 → IMG_SIZE → crop to input_size → orig_size
        ih, iw = self._input_size
        oh, ow = self._orig_size
        stacked = low_res.transpose(1, 2, 0)           # (256, 256, 3)
        m = cv2.resize(stacked, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        m = m[:ih, :iw]
        m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_LINEAR)
        if m.ndim == 2:                                 # single-point edge case
            m = m[:, :, np.newaxis]
        masks = (m > 0).transpose(2, 0, 1)             # (3, oh, ow) bool

        return masks, scores


class SAM2Predictor:
    """ONNX Runtime based SAM2 (Hiera) predictor.

    Matches the vietanhdev/samexporter ONNX export: a square-resize image
    encoder producing (high_res_feats_0, high_res_feats_1, image_embedding),
    and a prompt decoder consuming those plus point/mask prompts. Decoder
    I/O is bound positionally (not by name), matching that export's own
    inference code.
    """

    def __init__(self, encoder_path: str, decoder_path: str) -> None:
        import onnxruntime as ort

        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._enc = _make_session(encoder_path, providers)
        self._dec = _make_session(decoder_path, providers)

        enc_inputs = self._enc.get_inputs()
        self._enc_input_name   = enc_inputs[0].name
        self._enc_output_names = [o.name for o in self._enc.get_outputs()]
        enc_shape = enc_inputs[0].shape
        self._input_hw = (int(enc_shape[2]), int(enc_shape[3]))   # (ih, iw)

        self._dec_input_names  = [i.name for i in self._dec.get_inputs()]
        self._dec_output_names = [o.name for o in self._dec.get_outputs()]

        self._embeddings = None            # [high_res_feats_0, high_res_feats_1, image_embedding]
        self._orig_size: Tuple[int, int] = (0, 0)

    @property
    def device(self) -> str:
        p = self._enc.get_providers()
        return "CUDA" if "CUDAExecutionProvider" in p else "CPU"

    def set_image(self, image_rgb: np.ndarray) -> None:
        h, w = image_rgb.shape[:2]
        self._orig_size = (h, w)
        ih, iw = self._input_hw

        resized = cv2.resize(image_rgb, (iw, ih))
        x = resized.astype(np.float32) / 255.0
        x = (x - _SAM2_MEAN) / _SAM2_STD
        x = x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        self._embeddings = self._enc.run(self._enc_output_names, {self._enc_input_name: x})

    def predict(
        self,
        points: List[Tuple[int, int]],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (masks [K,H,W] bool, scores [K] float32)."""
        if self._embeddings is None:
            raise RuntimeError("set_image() must be called before predict()")

        ih, iw = self._input_hw
        oh, ow = self._orig_size

        coords = np.array(points, dtype=np.float32)
        coords[..., 0] = coords[..., 0] / ow * iw
        coords[..., 1] = coords[..., 1] / oh * ih
        coords = coords[np.newaxis]                       # [1, N, 2]
        lbls = np.array(labels, dtype=np.float32)[np.newaxis]  # [1, N]

        mask_input = np.zeros((lbls.shape[0], 1, ih // 4, iw // 4), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)

        # Encoder output order: high_res_feats_0, high_res_feats_1, image_embedding.
        # Decoder input order:  image_embedding, high_res_feats_0, high_res_feats_1,
        #                       point_coords, point_labels, mask_input, has_mask_input.
        hr0, hr1, image_embed = self._embeddings
        feed_values = [image_embed, hr0, hr1, coords, lbls, mask_input, has_mask_input]
        feed = dict(zip(self._dec_input_names, feed_values))

        out = self._dec.run(self._dec_output_names, feed)
        masks_raw = out[0][0]           # (K, ih, iw) logits
        scores = out[1].reshape(-1)     # (K,)

        k = masks_raw.shape[0]
        resized = np.stack([
            cv2.resize(masks_raw[i], (ow, oh), interpolation=cv2.INTER_LINEAR)
            for i in range(k)
        ])
        masks = resized > 0.0           # (K, oh, ow) bool

        return masks, scores.astype(np.float32)


MODEL_EDGESAM = "edgesam"
MODEL_SAM2 = "sam2"
MODEL_SAM2_BASE_PLUS = "sam2_base_plus"
MODEL_SAM2_LARGE = "sam2_large"
MODEL_SAM2_1_LARGE = "sam2_1_large"

MODEL_INFO = {
    MODEL_EDGESAM: {
        "label": "EdgeSAM",
        "encoder": ENCODER_PATH,
        "decoder": DECODER_PATH,
        "cls": EdgeSAMPredictor,
    },
    MODEL_SAM2: {
        "label": "SAM2 (Hiera-Tiny)",
        "encoder": SAM2_ENCODER_PATH,
        "decoder": SAM2_DECODER_PATH,
        "cls": SAM2Predictor,
        "hidden": True,
    },
    MODEL_SAM2_BASE_PLUS: {
        "label": "SAM2 (Hiera-Base+)",
        "encoder": SAM2_BASE_PLUS_ENCODER_PATH,
        "decoder": SAM2_BASE_PLUS_DECODER_PATH,
        "cls": SAM2Predictor,
        "hidden": True,
    },
    MODEL_SAM2_LARGE: {
        "label": "SAM2 (Hiera-Large)",
        "encoder": SAM2_LARGE_ENCODER_PATH,
        "decoder": SAM2_LARGE_DECODER_PATH,
        "cls": SAM2Predictor,
    },
    MODEL_SAM2_1_LARGE: {
        "label": "SAM2.1 (Hiera-Large)",
        "encoder": SAM2_1_LARGE_ENCODER_PATH,
        "decoder": SAM2_1_LARGE_DECODER_PATH,
        "cls": SAM2Predictor,
    },
}


def visible_models() -> List[str]:
    """Model keys to offer in the UI, in MODEL_INFO order (hidden ones excluded)."""
    return [key for key, info in MODEL_INFO.items() if not info.get("hidden")]


def missing_weights(model_key: str) -> List[str]:
    info = MODEL_INFO[model_key]
    return [p for p in (info["encoder"], info["decoder"]) if not os.path.exists(p)]


def create_predictor(model_key: str):
    info = MODEL_INFO[model_key]
    return info["cls"](info["encoder"], info["decoder"])
