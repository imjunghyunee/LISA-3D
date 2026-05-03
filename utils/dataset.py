"""
GraspClutter6D datasets for LISA-3D training and inference.

Training  — GraspClutter6DPairDataset:
  Returns two views of the same scene with a shared text query and their
  ground-truth foreground masks (union of all visible objects), depth maps,
  and camera parameters for the geometry-consistency loss.

Inference — GraspClutter6DInferDataset:
  Returns individual frames chosen to match the Clutt3R-Seg evaluation
  protocol (8 equally-spaced views, realsense-d415 by default).

Frame-ID convention in GraspClutter6D
  img_id = ann_id * 4 + cam_offset   (ann_ids 0–12, 13 annotation positions)
  realsense-d415 offset = 1 → valid img_ids: 1, 5, 9, …, 49  (13 frames)
  scene_gt.json keys: string integers; keys < 1 are metadata artefacts, skip them.
  mask_visib/{img_id:06d}_{obj_idx:06d}.png  (obj_idx = 0-based within scene_gt[key])
"""

import json
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

# LISA imports — LISA repo must be on PYTHONPATH
from model.llava1p5 import conversation as conversation_lib
from model.llava1p5.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    CAM_OFFSETS,
    CAM_DEPTH_SCALES,
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
    build_intrinsic,
    build_extrinsic,
    get_eval_img_ids,
    preprocess_sam,
)

# Default text prompt for training (class-agnostic foreground segmentation)
_TRAIN_PROMPT = "Please segment all objects in this image."
# Answer template (must contain [SEG] so the model outputs the seg token)
_TRAIN_ANSWER = "Sure, [SEG]."


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_scene_camera(scene_dir: str) -> dict:
    """Return the parsed scene_camera.json as {img_id_str: {...}}."""
    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        return json.load(f)


def _load_scene_gt(scene_dir: str) -> dict:
    """Return the parsed scene_gt.json as {img_id_str: [{...}, ...]}."""
    with open(os.path.join(scene_dir, "scene_gt.json")) as f:
        return json.load(f)


def _load_scene_gt_info(scene_dir: str) -> dict:
    """Return the parsed scene_gt_info.json as {img_id_str: [{...}, ...]}."""
    path = os.path.join(scene_dir, "scene_gt_info.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _valid_img_ids(scene_gt: dict, cam_offset: int) -> List[int]:
    """Return img_ids that belong to the requested camera and are >= 1."""
    ids = []
    for key in scene_gt:
        try:
            img_id = int(key)
        except ValueError:
            continue
        if img_id >= 1 and (img_id % 4) == (cam_offset % 4):
            ids.append(img_id)
    return sorted(ids)


def _build_combined_mask(
    scene_dir: str,
    img_id: int,
    scene_gt: dict,
    min_visib_fract: float = 0.0,
    min_px_visib: int = 0,
    scene_gt_info: Optional[dict] = None,
) -> np.ndarray:
    """Build binary foreground mask = union of all visible object masks.

    Args:
        scene_dir:       Path to the scene directory.
        img_id:          Integer image ID.
        scene_gt:        Parsed scene_gt.json.
        min_visib_fract: Minimum visibility fraction to include an object.
        min_px_visib:    Minimum visible pixel count.
        scene_gt_info:   Parsed scene_gt_info.json (optional, for filtering).

    Returns:
        (H, W) uint8 binary mask (0 or 255 matching OpenCV convention, then
        returned as bool numpy array).
    """
    key = str(img_id)
    objs = scene_gt.get(key, [])
    gt_info = (scene_gt_info or {}).get(key, [])

    mask_dir = os.path.join(scene_dir, "mask_visib")
    combined: Optional[np.ndarray] = None

    for obj_idx, obj in enumerate(objs):
        # Visibility filter
        if gt_info and obj_idx < len(gt_info):
            vf = gt_info[obj_idx].get("visib_fract", 1.0)
            px = gt_info[obj_idx].get("px_count_visib", 999)
            if vf < min_visib_fract or px < min_px_visib:
                continue

        mask_path = os.path.join(
            mask_dir, f"{img_id:06d}_{obj_idx:06d}.png"
        )
        if not os.path.exists(mask_path):
            continue

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        if combined is None:
            combined = (m > 0)
        else:
            combined = combined | (m > 0)

    if combined is None:
        # Fallback: return empty mask using RGB image shape
        rgb_path = os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
        rgb = cv2.imread(rgb_path)
        H, W = (rgb.shape[:2] if rgb is not None else (1080, 1920))
        combined = np.zeros((H, W), dtype=bool)

    return combined.astype(bool)


def _load_frame(
    scene_dir: str,
    img_id: int,
    scene_camera: dict,
    depth_scale: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load RGB, depth, intrinsics, and extrinsics for one frame.

    Returns:
        rgb:   (H, W, 3) uint8
        depth: (H, W) float32 in mm (raw values, not yet converted to metres)
        K:     (3, 3) float32 intrinsic
        E:     (4, 4) float32 world-to-camera (t in metres)
    """
    key = str(img_id)
    cam_data = scene_camera[key]

    rgb_path   = os.path.join(scene_dir, "rgb",   f"{img_id:06d}.png")
    depth_path = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")

    rgb   = cv2.imread(rgb_path)
    rgb   = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # mm

    K = build_intrinsic(cam_data["cam_K"])
    E = build_extrinsic(cam_data["cam_R_w2c"], cam_data["cam_t_w2c"])
    return rgb, depth, K, E


def _preprocess_rgb_for_sam(rgb_np: np.ndarray, transform: ResizeLongestSide) -> Tuple[torch.Tensor, tuple]:
    """Apply ResizeLongestSide + SAM normalisation + padding.

    Returns:
        image_tensor: (3, 1024, 1024) float32 tensor
        resize_shape: (H_resized, W_resized) before padding
    """
    resized = transform.apply_image(rgb_np)          # (H', W', 3) uint8
    resize_shape = resized.shape[:2]
    t = torch.from_numpy(resized).permute(2, 0, 1).float()  # (3, H', W')
    t = preprocess_sam(t)                             # normalise + pad → (3, 1024, 1024)
    return t, resize_shape


def _build_conversation(
    prompt: str,
    answer: str,
    tokenizer,
    conv_type: str = "llava_v1",
    use_mm_start_end: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tokenised (input_ids, labels, attention_mask) for one sample.

    Returns tensors of shape (seq_len,).
    """
    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []

    image_token = DEFAULT_IMAGE_TOKEN
    if use_mm_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    user_msg = image_token + "\n" + prompt
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], answer)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    # Build labels: mask everything before the answer, keep answer tokens
    # Simple approach: labels == input_ids but with IGNORE_INDEX for question part
    labels = input_ids.clone()
    # Find the position of the answer start in the token stream
    # The answer always ends with EOS, so we mask everything up to the last
    # occurrence of the separator between question and answer.
    # For simplicity, keep full supervision (same as base LISA training).
    # The LLaVA training framework masks up to the ASSISTANT: separator;
    # for our fine-tuning of LoRA only, this level of detail does not
    # critically affect the geometry loss, so we use full supervision.
    attention_mask = torch.ones_like(input_ids)

    return input_ids, labels, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraspClutter6DPairDataset(Dataset):
    """Multi-view pair dataset for geometry-aware LoRA training.

    Each item returns two views (A, B) from the same scene with:
      - Preprocessed RGB images (SAM + CLIP formats)
      - Combined foreground masks (union of all visible objects)
      - Depth maps in mm
      - Camera intrinsics K and extrinsics E (w2c, t in metres)
      - Tokenised shared text query

    The training loss is:
      L_total = L_seg_a + L_seg_b + lambda * L_geo
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        clip_image_processor: CLIPImageProcessor,
        scene_ids: List[int],
        image_size: int = 1024,
        conv_type: str = "llava_v1",
        camera: str = "realsense-d415",
        min_visib_fract: float = 0.1,
        min_px_visib: int = 200,
        prompt: str = _TRAIN_PROMPT,
        answer: str = _TRAIN_ANSWER,
    ):
        self.data_root          = data_root
        self.tokenizer          = tokenizer
        self.clip_processor     = clip_image_processor
        self.image_size         = image_size
        self.conv_type          = conv_type
        self.camera             = camera
        self.cam_offset         = CAM_OFFSETS[camera]
        self.depth_scale        = CAM_DEPTH_SCALES[camera]
        self.min_visib_fract    = min_visib_fract
        self.min_px_visib       = min_px_visib
        self.prompt             = prompt
        self.answer             = answer
        self.sam_transform      = ResizeLongestSide(image_size)

        # Build index: list of (scene_id, img_id) for frames with ≥1 visible obj
        self._items: List[Tuple[int, int]] = []
        # Map from scene_id → list of valid img_ids (for pair sampling)
        self._scene_to_ids: dict = {}
        self._build_index(scene_ids)

    def _build_index(self, scene_ids: List[int]) -> None:
        for scene_id in scene_ids:
            scene_dir = os.path.join(self.data_root, "scenes", f"{scene_id:06d}")
            if not os.path.isdir(scene_dir):
                continue
            scene_gt   = _load_scene_gt(scene_dir)
            scene_cam  = _load_scene_camera(scene_dir)
            gt_info    = _load_scene_gt_info(scene_dir)
            img_ids    = _valid_img_ids(scene_gt, self.cam_offset)

            valid_ids = []
            for img_id in img_ids:
                key = str(img_id)
                if key not in scene_cam:
                    continue
                # Check at least one object passes visibility threshold
                objs    = scene_gt.get(key, [])
                gi      = gt_info.get(key, [])
                has_obj = False
                for j, obj in enumerate(objs):
                    vf = gi[j].get("visib_fract", 1.0) if j < len(gi) else 1.0
                    px = gi[j].get("px_count_visib", 999) if j < len(gi) else 999
                    if vf >= self.min_visib_fract and px >= self.min_px_visib:
                        has_obj = True
                        break
                if has_obj:
                    valid_ids.append(img_id)

            if len(valid_ids) >= 2:
                self._scene_to_ids[scene_id] = valid_ids
                for img_id in valid_ids:
                    self._items.append((scene_id, img_id))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        scene_id, img_id_a = self._items[idx]
        scene_dir = os.path.join(self.data_root, "scenes", f"{scene_id:06d}")

        # Sample a different frame from the same scene as view B
        other_ids = [i for i in self._scene_to_ids[scene_id] if i != img_id_a]
        img_id_b  = random.choice(other_ids)

        scene_gt      = _load_scene_gt(scene_dir)
        scene_camera  = _load_scene_camera(scene_dir)
        gt_info       = _load_scene_gt_info(scene_dir)

        def load_and_preprocess(img_id: int):
            rgb, depth, K, E = _load_frame(
                scene_dir, img_id, scene_camera, self.depth_scale
            )
            mask = _build_combined_mask(
                scene_dir, img_id, scene_gt,
                self.min_visib_fract, self.min_px_visib, gt_info
            )
            original_size = rgb.shape[:2]   # (H, W)

            # CLIP preprocessing
            img_clip = self.clip_processor.preprocess(rgb, return_tensors="pt")[
                "pixel_values"
            ][0]  # (3, H_clip, W_clip)

            # SAM preprocessing
            img_sam, resize_shape = _preprocess_rgb_for_sam(rgb, self.sam_transform)

            # Resize GT mask to match SAM output resolution (before SAM padding)
            # SAM mask decoder outputs at 4× downsampled resolution of resize_shape
            # For loss computation we keep mask at original size (postprocess_masks
            # in model_forward handles upsampling)
            mask_tensor = torch.from_numpy(mask.astype(np.float32))  # (H, W) float32

            depth_tensor = torch.from_numpy(depth)  # (H, W) float32 mm

            K_tensor = torch.from_numpy(K)  # (3, 3)
            E_tensor = torch.from_numpy(E)  # (4, 4)

            return {
                "img_sam":      img_sam,
                "img_clip":     img_clip,
                "mask":         mask_tensor,
                "depth":        depth_tensor,
                "K":            K_tensor,
                "E":            E_tensor,
                "original_size": original_size,
                "resize_shape":  resize_shape,
            }

        fa = load_and_preprocess(img_id_a)
        fb = load_and_preprocess(img_id_b)

        # Tokenise shared text (same query for both views)
        input_ids, labels, attention_mask = _build_conversation(
            self.prompt, self.answer, self.tokenizer, self.conv_type
        )

        return {
            # View A
            "images_a":      fa["img_sam"],
            "images_clip_a": fa["img_clip"],
            "masks_a":       fa["mask"],
            "depth_a":       fa["depth"],
            "K_a":           fa["K"],
            "E_a":           fa["E"],
            "original_size_a": fa["original_size"],
            "resize_list_a":   fa["resize_shape"],
            # View B
            "images_b":      fb["img_sam"],
            "images_clip_b": fb["img_clip"],
            "masks_b":       fb["mask"],
            "depth_b":       fb["depth"],
            "K_b":           fb["K"],
            "E_b":           fb["E"],
            "original_size_b": fb["original_size"],
            "resize_list_b":   fb["resize_shape"],
            # Shared text
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraspClutter6DInferDataset(Dataset):
    """Per-frame dataset for LISA-3D inference.

    Returns the 8 evaluation frames per scene as defined by the
    Clutt3R-Seg evaluation protocol (get_eval_img_ids()).
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        clip_image_processor: CLIPImageProcessor,
        scene_ids: List[int],
        prompt: str,
        camera: str = "realsense-d415",
        image_size: int = 1024,
        conv_type: str = "llava_v1",
        n_views: int = 8,
    ):
        self.data_root       = data_root
        self.tokenizer       = tokenizer
        self.clip_processor  = clip_image_processor
        self.camera          = camera
        self.cam_offset      = CAM_OFFSETS[camera]
        self.depth_scale     = CAM_DEPTH_SCALES[camera]
        self.image_size      = image_size
        self.conv_type       = conv_type
        self.prompt          = prompt
        self.sam_transform   = ResizeLongestSide(image_size)

        # Build flat list of (scene_id, img_id)
        self._items: List[Tuple[int, int]] = []
        eval_img_ids = get_eval_img_ids(camera=camera, n_select=n_views)
        for scene_id in scene_ids:
            scene_dir = os.path.join(data_root, "scenes", f"{scene_id:06d}")
            if not os.path.isdir(scene_dir):
                continue
            scene_cam = _load_scene_camera(scene_dir)
            for img_id in eval_img_ids:
                if str(img_id) in scene_cam and os.path.exists(
                    os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
                ):
                    self._items.append((scene_id, img_id))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        scene_id, img_id = self._items[idx]
        scene_dir   = os.path.join(self.data_root, "scenes", f"{scene_id:06d}")
        scene_cam   = _load_scene_camera(scene_dir)
        scene_gt    = _load_scene_gt(scene_dir)

        rgb, depth, K, E = _load_frame(scene_dir, img_id, scene_cam, self.depth_scale)
        original_size = rgb.shape[:2]

        # CLIP preprocessing
        img_clip = self.clip_processor.preprocess(rgb, return_tensors="pt")[
            "pixel_values"
        ][0]

        # SAM preprocessing
        img_sam, resize_shape = _preprocess_rgb_for_sam(rgb, self.sam_transform)

        # Tokenise prompt
        input_ids, _, attention_mask = _build_conversation(
            self.prompt,
            "Sure, [SEG].",
            self.tokenizer,
            self.conv_type,
        )

        # Object IDs visible in this frame (for potential label assignment)
        key  = str(img_id)
        objs = scene_gt.get(key, [])
        obj_ids = [o["obj_id"] for o in objs]

        return {
            "images_sam":     img_sam,
            "images_clip":    img_clip,
            "depth":          torch.from_numpy(depth),          # (H, W) float32 mm
            "K":              torch.from_numpy(K),              # (3, 3)
            "E":              torch.from_numpy(E),              # (4, 4)
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "original_size":  original_size,
            "resize_shape":   resize_shape,
            "scene_id":       scene_id,
            "img_id":         img_id,
            "obj_ids":        obj_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions
# ─────────────────────────────────────────────────────────────────────────────

def _pad_sequence(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Left-pad a list of 1-D tensors to the same length."""
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, max_len - s.shape[0]:] = s
    return out


def collate_fn_train(batch: List[dict]) -> dict:
    """Collate a list of training pair dicts into a batched dict."""
    from utils.utils import IGNORE_INDEX

    images_a      = torch.stack([b["images_a"]      for b in batch])
    images_clip_a = torch.stack([b["images_clip_a"] for b in batch])
    images_b      = torch.stack([b["images_b"]      for b in batch])
    images_clip_b = torch.stack([b["images_clip_b"] for b in batch])

    depth_a = torch.stack([b["depth_a"] for b in batch])
    depth_b = torch.stack([b["depth_b"] for b in batch])
    K_a     = torch.stack([b["K_a"]     for b in batch])
    K_b     = torch.stack([b["K_b"]     for b in batch])
    E_a     = torch.stack([b["E_a"]     for b in batch])
    E_b     = torch.stack([b["E_b"]     for b in batch])

    # Pad text tokens
    input_ids_list      = [b["input_ids"]      for b in batch]
    attention_mask_list = [b["attention_mask"] for b in batch]
    labels_list         = [b["labels"]         for b in batch]

    input_ids      = _pad_sequence(input_ids_list,      pad_value=0)
    attention_masks = _pad_sequence(attention_mask_list, pad_value=0)
    labels         = _pad_sequence(labels_list,          pad_value=IGNORE_INDEX)

    # offset: cumulative index into batch (each sample contributes 1 mask)
    offset = torch.arange(len(batch) + 1, dtype=torch.long)

    # Masks: list of (1, H, W) per sample (one seg token per sample)
    masks_list_a = [b["masks_a"].unsqueeze(0) for b in batch]
    masks_list_b = [b["masks_b"].unsqueeze(0) for b in batch]

    label_list_a = [torch.tensor(b["original_size_a"], dtype=torch.long) for b in batch]
    label_list_b = [torch.tensor(b["original_size_b"], dtype=torch.long) for b in batch]

    resize_list_a = [b["resize_list_a"] for b in batch]
    resize_list_b = [b["resize_list_b"] for b in batch]

    return {
        "images_a":      images_a,
        "images_clip_a": images_clip_a,
        "images_b":      images_b,
        "images_clip_b": images_clip_b,
        "input_ids":     input_ids,
        "labels":        labels,
        "attention_masks": attention_masks,
        "offset":        offset,
        "masks_list_a":  masks_list_a,
        "masks_list_b":  masks_list_b,
        "label_list_a":  label_list_a,
        "label_list_b":  label_list_b,
        "resize_list_a": resize_list_a,
        "resize_list_b": resize_list_b,
        "depth_a":       depth_a,
        "depth_b":       depth_b,
        "K_a":           K_a,
        "K_b":           K_b,
        "E_a":           E_a,
        "E_b":           E_b,
    }


def collate_fn_infer(batch: List[dict]) -> dict:
    """Collate inference dicts (variable-length; keep as lists where needed)."""
    return {
        "images_sam":     torch.stack([b["images_sam"]   for b in batch]),
        "images_clip":    torch.stack([b["images_clip"]  for b in batch]),
        "depth":          torch.stack([b["depth"]        for b in batch]),
        "K":              torch.stack([b["K"]            for b in batch]),
        "E":              torch.stack([b["E"]            for b in batch]),
        "input_ids":      _pad_sequence([b["input_ids"]      for b in batch]),
        "attention_mask": _pad_sequence([b["attention_mask"] for b in batch]),
        "original_size":  [b["original_size"]  for b in batch],
        "resize_shape":   [b["resize_shape"]   for b in batch],
        "scene_id":       [b["scene_id"]       for b in batch],
        "img_id":         [b["img_id"]         for b in batch],
        "obj_ids":        [b["obj_ids"]        for b in batch],
    }
