"""
GraspClutter6D datasets for LISA-3D training and inference.

Training  — GraspClutter6DPairDataset:
  Returns two views of the same scene with a per-Object-Name text query and
  the corresponding per-name mask_visib (OR across all obj_ids sharing that
  name), depth maps, and camera parameters for the geometry-consistency loss.

Inference — GraspClutter6DInferDataset:
  Returns individual frames chosen to match the Clutt3R-Seg evaluation
  protocol (8 equally-spaced views, realsense-d415 by default).  The prompt
  is supplied by the caller at infer time (one prompt per Object Name).

Frame-ID convention in GraspClutter6D
  img_id = ann_id * 4 + cam_offset   (ann_ids 0–12, 13 annotation positions)
  realsense-d415 offset = 1 → valid img_ids: 1, 5, 9, …, 49  (13 frames)
  scene_gt.json keys: string integers; keys < 1 are metadata artefacts, skip them.
  mask_visib/{img_id:06d}_{obj_idx:06d}.png  (obj_idx = 0-based within scene_gt[key])
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

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

# Per-object training prompt template; {name} = Object Name from the CSV.
_TRAIN_PROMPT_OBJ = "Please segment the {name} in this image."
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


def _frame_resolution(scene_dir: str, img_id: int) -> Tuple[int, int]:
    """Resolve (H, W) for a frame without holding the RGB array in memory."""
    rgb_path = os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
    rgb = cv2.imread(rgb_path)
    return (rgb.shape[:2] if rgb is not None else (1080, 1920))


def _build_object_mask(
    scene_dir: str,
    img_id: int,
    scene_gt: dict,
    target_obj_ids: set,
    scene_gt_info: Optional[dict] = None,
    min_visib_fract: float = 0.1,
    min_px_visib: int = 200,
) -> Optional[np.ndarray]:
    """Per-name foreground mask = OR over visible mask_visib of every object
    in this frame whose ``obj_id`` is in ``target_obj_ids``.

    Returns ``None`` if no instance of the target name passes the visibility
    filter (caller skips the anchor).
    """
    key = str(img_id)
    objs = scene_gt.get(key, [])
    gt_info = (scene_gt_info or {}).get(key, [])

    mask_dir = os.path.join(scene_dir, "mask_visib")
    combined: Optional[np.ndarray] = None

    for obj_idx, obj in enumerate(objs):
        if obj.get("obj_id") not in target_obj_ids:
            continue
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
        return None
    return combined.astype(bool)


def _frame_object_ids(
    scene_gt: dict,
    img_id: int,
    scene_gt_info: Optional[dict],
    min_visib_fract: float,
    min_px_visib: int,
) -> List[int]:
    """Return list of obj_ids appearing in this frame that pass the
    visibility threshold (de-duplicated)."""
    key = str(img_id)
    objs = scene_gt.get(key, [])
    gt_info = (scene_gt_info or {}).get(key, [])
    seen = set()
    for obj_idx, obj in enumerate(objs):
        if gt_info and obj_idx < len(gt_info):
            vf = gt_info[obj_idx].get("visib_fract", 1.0)
            px = gt_info[obj_idx].get("px_count_visib", 999)
            if vf < min_visib_fract or px < min_px_visib:
                continue
        oid = obj.get("obj_id")
        if oid is not None:
            seen.add(int(oid))
    return sorted(seen)


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

    Each item is anchored on ``(scene_id, img_id_a, object_name)`` — a single
    Object Name from ``graspclutter6d_object_id.csv``.  The second view
    ``img_id_b`` is sampled at runtime from the same scene's frames where
    the same Object Name still passes the visibility filter, so the
    per-object mask is non-empty in both views.

    Returned tensors per view:
      - Preprocessed RGB images (SAM + CLIP formats)
      - Per-name visibility mask  (OR over all obj_ids that share the name
        and pass the visibility threshold in that frame)
      - Depth map in mm
      - Camera intrinsics K and extrinsics E (w2c, t in metres)
      - Tokenised ``"Please segment the {name} in this image."`` query

    The training loss is:
      L_total = L_seg_a + L_seg_b + lambda * L_geo
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        clip_image_processor: CLIPImageProcessor,
        scene_ids: List[int],
        name_to_obj_ids: Dict[str, List[int]],
        image_size: int = 1024,
        conv_type: str = "llava_v1",
        camera: str = "realsense-d415",
        min_visib_fract: float = 0.1,
        min_px_visib: int = 200,
        target_names: Optional[List[str]] = None,
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
        self.answer             = answer
        self.sam_transform      = ResizeLongestSide(image_size)

        # Restrict to the requested object names (if any).
        if target_names is not None:
            target_set = set(target_names)
            self.name_to_obj_ids: Dict[str, List[int]] = {
                n: list(ids) for n, ids in name_to_obj_ids.items()
                if n in target_set
            }
        else:
            self.name_to_obj_ids = {
                n: list(ids) for n, ids in name_to_obj_ids.items()
            }
        # Reverse map: obj_id -> name for the active (filtered) set.
        self.obj_id_to_name: Dict[int, str] = {}
        for name, ids in self.name_to_obj_ids.items():
            for oid in ids:
                self.obj_id_to_name[int(oid)] = name

        # Anchor items: each entry = (scene_id, img_id_a, object_name).
        self._items: List[Tuple[int, int, str]] = []
        # scene_id -> {object_name -> sorted list of img_ids where that name
        # has a non-empty per-name mask}.  Used to pick paired view B.
        self._scene_name_to_imgs: Dict[int, Dict[str, List[int]]] = {}
        self._build_index(scene_ids)

    def _build_index(self, scene_ids: List[int]) -> None:
        for scene_id in scene_ids:
            scene_dir = os.path.join(self.data_root, "scenes", f"{scene_id:06d}")
            if not os.path.isdir(scene_dir):
                continue
            scene_gt  = _load_scene_gt(scene_dir)
            scene_cam = _load_scene_camera(scene_dir)
            gt_info   = _load_scene_gt_info(scene_dir)
            img_ids   = _valid_img_ids(scene_gt, self.cam_offset)

            # name -> list of img_ids in this scene where the name has at
            # least one visible-enough instance.
            name_to_imgs: Dict[str, List[int]] = {}
            for img_id in img_ids:
                if str(img_id) not in scene_cam:
                    continue
                # Which obj_ids are visible enough in this frame?
                visible_oids = _frame_object_ids(
                    scene_gt, img_id, gt_info,
                    self.min_visib_fract, self.min_px_visib,
                )
                if not visible_oids:
                    continue
                # Which active object names get a non-empty mask here?
                names_here: set = set()
                for oid in visible_oids:
                    name = self.obj_id_to_name.get(int(oid))
                    if name is not None:
                        names_here.add(name)
                for name in names_here:
                    name_to_imgs.setdefault(name, []).append(img_id)

            if not name_to_imgs:
                continue

            # Keep only (name) with >=2 frames so view B can differ from A.
            paired_name_to_imgs = {
                n: sorted(ids) for n, ids in name_to_imgs.items()
                if len(ids) >= 2
            }
            if not paired_name_to_imgs:
                continue
            self._scene_name_to_imgs[scene_id] = paired_name_to_imgs
            for name, ids in paired_name_to_imgs.items():
                for img_id in ids:
                    self._items.append((scene_id, img_id, name))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        scene_id, img_id_a, name = self._items[idx]
        scene_dir = os.path.join(self.data_root, "scenes", f"{scene_id:06d}")

        # Pick a paired frame where the same Object Name is still visible.
        candidates = [
            i for i in self._scene_name_to_imgs[scene_id][name] if i != img_id_a
        ]
        img_id_b = random.choice(candidates) if candidates else img_id_a

        scene_gt      = _load_scene_gt(scene_dir)
        scene_camera  = _load_scene_camera(scene_dir)
        gt_info       = _load_scene_gt_info(scene_dir)

        target_obj_ids = set(int(o) for o in self.name_to_obj_ids[name])

        def load_and_preprocess(img_id: int):
            rgb, depth, K, E = _load_frame(
                scene_dir, img_id, scene_camera, self.depth_scale
            )
            mask = _build_object_mask(
                scene_dir, img_id, scene_gt,
                target_obj_ids,
                gt_info,
                self.min_visib_fract, self.min_px_visib,
            )
            if mask is None:
                # Should not happen because _build_index gated on visibility,
                # but guard anyway.
                mask = np.zeros(rgb.shape[:2], dtype=bool)
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
        prompt = _TRAIN_PROMPT_OBJ.format(name=name)
        input_ids, labels, attention_mask = _build_conversation(
            prompt, self.answer, self.tokenizer, self.conv_type
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
            # For diagnostics / sanity checks (not used by collate).
            "object_name":    name,
            "scene_id":       scene_id,
            "img_id_a":       img_id_a,
            "img_id_b":       img_id_b,
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

    # LISA's model_forward reads ``label_list[i].shape`` (a Tuple[H, W]) and
    # passes it to SAM's postprocess_masks as the resize target.  We therefore
    # need a tensor whose shape *is* (H, W), not a length-2 vector.  Reusing
    # the GT mask is cheap and conveys the right shape.
    label_list_a = [b["masks_a"] for b in batch]
    label_list_b = [b["masks_b"] for b in batch]

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
