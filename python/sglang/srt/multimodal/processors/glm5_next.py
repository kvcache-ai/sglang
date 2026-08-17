"""Self-contained image processor for GLM-5-Next.

The pinned transformers-kt wheel has the reusable GLM-4.6V tokenizer and
patchifier, but it does not register the checkpoint's Glmga image processor.
Keep the small GLM-5 resize difference here instead of requiring a patched
Transformers checkout at runtime.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import SizeDict
from transformers.models.glm46v.image_processing_glm46v import (
    Glm46VImageProcessor,
    smart_resize,
)
from transformers.models.glm46v.processing_glm46v import Glm46VProcessor

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor


GLM5_NEXT_MIN_PIXELS = 12_544
GLM5_NEXT_DEFAULT_MAX_PIXELS = 1_254_400
GLM5_NEXT_CHECKPOINT_MAX_PIXELS = 9_633_792
GLM5_NEXT_PATCH_EXPAND_FACTOR = 2
_GLM5_NEXT_IMAGE_CONFIG_KEYS = frozenset(
    {
        "do_rescale",
        "image_mean",
        "image_processor_type",
        "image_std",
        "merge_size",
        "patch_size",
        "patch_expand_factor",
        "size",
        "temporal_patch_size",
    }
)
_GLM5_NEXT_IMAGE_CONFIG_VALUES = {
    "do_rescale": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_processor_type": "GlmgaImageProcessor",
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "merge_size": 2,
    "patch_size": 14,
    "patch_expand_factor": GLM5_NEXT_PATCH_EXPAND_FACTOR,
    "temporal_patch_size": 2,
}


class Glm5NextImageProcessor(Glm46VImageProcessor):
    """GLM-4.6V patchification with GLM-5's expanded resize factor."""

    size = {
        "shortest_edge": GLM5_NEXT_MIN_PIXELS,
        "longest_edge": GLM5_NEXT_DEFAULT_MAX_PIXELS,
    }
    patch_expand_factor = GLM5_NEXT_PATCH_EXPAND_FACTOR

    def __init__(self, patch_expand_factor: int = 2, **kwargs) -> None:
        if patch_expand_factor != GLM5_NEXT_PATCH_EXPAND_FACTOR:
            raise ValueError(
                "GLM-5-Next requires patch_expand_factor=2; "
                f"got {patch_expand_factor!r}."
            )
        self.patch_expand_factor = patch_expand_factor
        super().__init__(**kwargs)

    @classmethod
    def from_checkpoint_config(cls, image_config: dict[str, Any]):
        config = dict(image_config)
        actual_keys = set(config)
        missing_keys = sorted(_GLM5_NEXT_IMAGE_CONFIG_KEYS - actual_keys)
        unknown_keys = sorted(actual_keys - _GLM5_NEXT_IMAGE_CONFIG_KEYS)
        if missing_keys or unknown_keys:
            raise ValueError(
                "GLM-5-Next pinned image processor metadata keys changed: "
                f"missing={missing_keys}, unknown={unknown_keys}."
            )
        for key, expected in _GLM5_NEXT_IMAGE_CONFIG_VALUES.items():
            actual = config[key]
            if actual != expected:
                raise ValueError(
                    f"GLM-5-Next pinned processor requires {key}={expected!r}; "
                    f"got {actual!r}."
                )

        checkpoint_size = config["size"]
        if not isinstance(checkpoint_size, dict) or checkpoint_size != {
            "shortest_edge": GLM5_NEXT_MIN_PIXELS,
            "longest_edge": GLM5_NEXT_CHECKPOINT_MAX_PIXELS,
        }:
            raise ValueError(
                "GLM-5-Next checkpoint processor size metadata changed: "
                f"got {checkpoint_size!r}."
            )
        config.pop("image_processor_type")
        config["size"] = {
            "shortest_edge": GLM5_NEXT_MIN_PIXELS,
            "longest_edge": GLM5_NEXT_DEFAULT_MAX_PIXELS,
        }
        return cls(**config)

    def _preprocess(
        self,
        images,
        do_resize,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        disable_grouping,
        return_tensors,
        **kwargs,
    ):
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(
                images, disable_grouping=disable_grouping
            )
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size * self.patch_expand_factor,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                resized_images_grouped[shape] = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            images = reorder_images(resized_images_grouped, grouped_images_index)

        return super()._preprocess(
            images=images,
            do_resize=False,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
            **kwargs,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = images_kwargs.get("size", self.size)
        min_pixels = (
            size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        )
        max_pixels = (
            size["longest_edge"] if isinstance(size, dict) else size.longest_edge
        )
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=patch_size * merge_size * self.patch_expand_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


class Glm5NextProcessor(Glm46VProcessor):
    """Image-only facade over the pinned GLM-4.6V token expansion logic."""

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if videos is not None:
            raise ValueError("GLM-5-Next Session D does not support video input.")
        return super().__call__(images=images, text=text, videos=None, **kwargs)


class Glm5NextSGLangProcessor(Glm4vImageProcessor):
    """Strict image-only request adapter registered only for GLM-5-Next."""

    models = [Glm5NextForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IMAGE_FACTOR = 56
        self.MIN_PIXELS = GLM5_NEXT_MIN_PIXELS
        self.MAX_PIXELS = self._resolve_max_pixels(server_args.mm_process_config)

        image_processor = getattr(_processor, "image_processor", None)
        if not isinstance(image_processor, Glm5NextImageProcessor):
            raise TypeError(
                "GLM-5-Next requires the built-in Glm5NextImageProcessor; "
                f"got {type(image_processor).__name__}."
            )
        if isinstance(image_processor.size, dict):
            image_processor.size["shortest_edge"] = self.MIN_PIXELS
            image_processor.size["longest_edge"] = self.MAX_PIXELS
        else:
            image_processor.size.shortest_edge = self.MIN_PIXELS
            image_processor.size.longest_edge = self.MAX_PIXELS

        # Rebuild after the GLM4V initializer so only the image modality is
        # advertised by this exact-model processor.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    @staticmethod
    def _resolve_max_pixels(mm_process_config) -> int:
        config = {} if mm_process_config is None else mm_process_config
        if not isinstance(config, dict):
            raise ValueError("GLM-5-Next --mm-process-config must be a JSON object.")
        unknown_root = set(config) - {"image"}
        if unknown_root:
            raise ValueError(
                "GLM-5-Next Session D accepts only the image processor config; "
                f"unsupported keys: {sorted(unknown_root)}."
            )
        image_config = config.get("image", {})
        if not isinstance(image_config, dict):
            raise ValueError("GLM-5-Next mm-process-config.image must be an object.")
        unknown_image = set(image_config) - {"max_pixels"}
        if unknown_image:
            raise ValueError(
                "GLM-5-Next accepts only image.max_pixels; unsupported keys: "
                f"{sorted(unknown_image)}."
            )
        max_pixels = image_config.get("max_pixels", GLM5_NEXT_DEFAULT_MAX_PIXELS)
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int):
            raise ValueError("GLM-5-Next image.max_pixels must be an integer.")
        if not GLM5_NEXT_MIN_PIXELS <= max_pixels <= GLM5_NEXT_CHECKPOINT_MAX_PIXELS:
            raise ValueError(
                "GLM-5-Next image.max_pixels must be within "
                f"[{GLM5_NEXT_MIN_PIXELS}, {GLM5_NEXT_CHECKPOINT_MAX_PIXELS}]; "
                f"got {max_pixels}."
            )
        return max_pixels

    def _count_image_placeholders(self, input_text) -> int:
        if isinstance(input_text, str):
            if self.VIDEO_TOKEN in input_text:
                raise ValueError("GLM-5-Next Session D does not support video input.")
            return input_text.count(self.IMAGE_TOKEN)
        if isinstance(input_text, (list, tuple)):
            if self.VIDEO_TOKEN_ID in input_text:
                raise ValueError("GLM-5-Next Session D does not support video input.")
            return input_text.count(self.IM_TOKEN_ID)
        raise TypeError("GLM-5-Next multimodal prompt must be text or token ids.")

    def _get_multi_image_mrope(
        self,
        input_ids,
        image_grid_thw,
        image_offsets,
        attention_mask,
    ):
        """Build GLM MRoPE positions while preserving adjacent image boundaries.

        The shared GLM4V helper discovers modality segments by grouping adjacent
        image-token IDs.  Two adjacent source placeholders therefore look like
        one image and consume only the first grid.  The request adapter already
        has one validated offset and one grid per source image, so use those
        explicit boundaries for the multi-image case.
        """

        if input_ids.ndim != 1:
            raise RuntimeError(
                "GLM-5-Next multi-image MRoPE expects one token sequence; "
                f"got shape={tuple(input_ids.shape)}."
            )
        sequence_length = int(input_ids.shape[0])
        if attention_mask is None:
            active_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            active_mask = attention_mask.reshape(-1).to(
                device=input_ids.device, dtype=torch.bool
            )
            if active_mask.numel() != sequence_length:
                raise RuntimeError(
                    "GLM-5-Next multi-image attention mask length changed: "
                    f"tokens={sequence_length}, mask={active_mask.numel()}."
                )

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        position_pieces = []
        assignment_pieces = []
        next_position = 0
        token_cursor = 0

        def append_text(indices):
            nonlocal next_position
            text_length = int(indices.numel())
            if text_length == 0:
                return
            text_positions = torch.arange(
                text_length, device=input_ids.device, dtype=input_ids.dtype
            )
            text_positions = text_positions.view(1, -1).expand(3, -1)
            text_positions = text_positions + next_position
            position_pieces.append(text_positions)
            assignment_pieces.append(indices)
            next_position = int(text_positions.max().item()) + 1

        for image_index, ((start, end), grid) in enumerate(
            zip(image_offsets, image_grid_thw)
        ):
            start, end = int(start), int(end)
            if start < token_cursor or end < start or end >= sequence_length:
                raise RuntimeError(
                    "GLM-5-Next multi-image offsets are not ordered in bounds at "
                    f"image {image_index}: start={start}, end={end}, "
                    f"cursor={token_cursor}, sequence_length={sequence_length}."
                )

            text_indices = active_indices[
                (active_indices >= token_cursor) & (active_indices < start)
            ]
            append_text(text_indices)

            image_indices = active_indices[
                (active_indices >= start) & (active_indices <= end)
            ]
            t, h, w = (int(value) for value in grid)
            merge_size = int(self.spatial_merge_size)
            if h % merge_size or w % merge_size:
                raise RuntimeError(
                    "GLM-5-Next image grid is not divisible by the spatial "
                    f"merge size at image {image_index}: grid={(t, h, w)}, "
                    f"merge_size={merge_size}."
                )
            grid_h, grid_w = h // merge_size, w // merge_size
            image_token_count = t * grid_h * grid_w
            if int(image_indices.numel()) != image_token_count:
                raise RuntimeError(
                    "GLM-5-Next multi-image offset/grid mismatch at image "
                    f"{image_index}: offset_tokens={image_indices.numel()}, "
                    f"grid_tokens={image_token_count}."
                )

            t_index = (
                torch.arange(t, device=input_ids.device, dtype=input_ids.dtype)
                .view(-1, 1)
                .expand(t, grid_h * grid_w)
                .reshape(-1)
            )
            h_index = (
                torch.arange(grid_h, device=input_ids.device, dtype=input_ids.dtype)
                .view(1, -1, 1)
                .expand(t, grid_h, grid_w)
                .reshape(-1)
            )
            w_index = (
                torch.arange(grid_w, device=input_ids.device, dtype=input_ids.dtype)
                .view(1, 1, -1)
                .expand(t, grid_h, grid_w)
                .reshape(-1)
            )
            image_positions = torch.stack((t_index, h_index, w_index)) + next_position
            position_pieces.append(image_positions)
            assignment_pieces.append(image_indices)
            next_position = int(image_positions.max().item()) + 1
            token_cursor = end + 1

        append_text(active_indices[active_indices >= token_cursor])
        assigned_indices = torch.cat(assignment_pieces)
        if not torch.equal(assigned_indices, active_indices):
            raise RuntimeError(
                "GLM-5-Next multi-image MRoPE segments do not cover exactly the "
                "active token sequence."
            )
        active_positions = torch.cat(position_pieces, dim=1)
        if active_positions.shape[1] != active_indices.numel():
            raise RuntimeError(
                "GLM-5-Next multi-image MRoPE position length changed: "
                f"positions={active_positions.shape[1]}, "
                f"active_tokens={active_indices.numel()}."
            )

        position_ids = torch.ones(
            (3, sequence_length),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        position_ids[:, active_indices] = active_positions
        position_delta = torch.tensor(
            [[next_position - sequence_length]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return position_ids, position_delta

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        del args, kwargs
        if audio_data:
            raise ValueError("GLM-5-Next Session D does not support audio input.")
        if getattr(request_obj, "video_data", None):
            raise ValueError("GLM-5-Next Session D does not support video input.")
        if not isinstance(image_data, list) or not image_data:
            count = len(image_data) if isinstance(image_data, list) else 0
            raise ValueError(
                "GLM-5-Next requires at least one image per image request; "
                f"got {count}."
            )
        image_count = len(image_data)
        if any(isinstance(image, dict) for image in image_data):
            raise ValueError(
                "GLM-5-Next does not accept processor_output or "
                "precomputed_embedding image inputs."
            )
        placeholder_count = self._count_image_placeholders(input_text)
        if placeholder_count != image_count:
            raise ValueError(
                "GLM-5-Next image count/placeholder count mismatch: "
                f"images={image_count}, placeholders={placeholder_count}."
            )

        # load_mm_data is synchronous: it submits image I/O to the processor's
        # executors and joins those futures before returning the organized
        # request payload.
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=None,
            multimodal_tokens=self.mm_tokens,
        )
        if (
            len(base_output.images) != image_count
            or base_output.videos
            or base_output.audios
        ):
            raise ValueError(
                "GLM-5-Next image loading changed the request cardinality: "
                f"expected={image_count}, loaded={len(base_output.images)}."
            )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        image_grid_thw = getattr(ret, "image_grid_thw", None)
        if image_grid_thw is None and isinstance(ret, dict):
            image_grid_thw = ret.get("image_grid_thw")
        expected_grid_shape = (image_count, 3)
        if image_grid_thw is None or tuple(image_grid_thw.shape) != expected_grid_shape:
            actual_grid_shape = (
                None if image_grid_thw is None else tuple(image_grid_thw.shape)
            )
            raise RuntimeError(
                "GLM-5-Next processor returned an invalid image_grid_thw shape: "
                f"expected={expected_grid_shape}, got={actual_grid_shape}."
            )
        if len(mm_items) != 1 or not mm_items[0].is_image():
            raise RuntimeError(
                "GLM-5-Next processor must return one bundled image item."
            )

        merge_area = self.spatial_merge_size**2
        patch_counts = [int(grid.prod().item()) for grid in image_grid_thw]
        if any(patch_count % merge_area for patch_count in patch_counts):
            raise RuntimeError(
                "GLM-5-Next image patch counts must be divisible by the spatial "
                f"merge area {merge_area}; got {patch_counts}."
            )
        expected_tokens_per_image = [
            patch_count // merge_area for patch_count in patch_counts
        ]
        expected_tokens = sum(expected_tokens_per_image)
        actual_tokens = int((input_ids == self.IM_TOKEN_ID).sum().item())
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                "GLM-5-Next image placeholder/feature mismatch: "
                f"expected {expected_tokens} image tokens, got {actual_tokens}."
            )

        source_offsets = mm_items[0].offsets or []
        covered_positions = []
        for start, end in source_offsets:
            start, end = int(start), int(end)
            if end < start:
                raise RuntimeError(
                    "GLM-5-Next processor returned an inverted image offset: "
                    f"start={start}, end={end}."
                )
            covered_positions.extend(range(start, end + 1))
        image_token_positions = (
            (input_ids == self.IM_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        )
        if covered_positions != image_token_positions:
            raise RuntimeError(
                "GLM-5-Next processor image offsets do not cover exactly the "
                "expanded image tokens."
            )

        # Adjacent placeholders form one token run in the generic processor.
        # Split by the per-image grid sizes so the scheduler still receives one
        # ordered offset per source image.
        normalized_offsets = []
        token_cursor = 0
        for image_index, expected_tokens_for_image in enumerate(
            expected_tokens_per_image
        ):
            positions = image_token_positions[
                token_cursor : token_cursor + expected_tokens_for_image
            ]
            token_cursor += expected_tokens_for_image
            is_contiguous = len(positions) == expected_tokens_for_image and all(
                right == left + 1 for left, right in zip(positions, positions[1:])
            )
            if not is_contiguous:
                raise RuntimeError(
                    "GLM-5-Next image offset/token mismatch at image "
                    f"{image_index}: expected={expected_tokens_for_image}, "
                    f"positions={positions}."
                )
            normalized_offsets.append((positions[0], positions[-1]))
        mm_items[0].offsets = normalized_offsets

        feature = mm_items[0].feature
        feature_shape = getattr(feature, "shape", None)
        if feature_shape is None:
            feature_shape = getattr(getattr(feature, "info_data", None), "shape", None)
        expected_feature_rows = sum(patch_counts)
        if (
            feature_shape is None
            or len(feature_shape) == 0
            or int(feature_shape[0]) != expected_feature_rows
        ):
            raise RuntimeError(
                "GLM-5-Next image feature/grid mismatch: "
                f"expected_rows={expected_feature_rows}, "
                f"feature_shape={None if feature_shape is None else tuple(feature_shape)}."
            )

        attention_mask = getattr(ret, "attention_mask", None)
        if attention_mask is None and isinstance(ret, dict):
            attention_mask = ret.get("attention_mask")
        if image_count == 1:
            mrope_positions, mrope_position_delta = (
                MRotaryEmbedding.get_rope_index_glm4v(
                    input_ids=input_ids.unsqueeze(0),
                    hf_config=self.hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
            )
            mrope_positions = mrope_positions.squeeze(1)
        else:
            mrope_positions, mrope_position_delta = self._get_multi_image_mrope(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                image_offsets=normalized_offsets,
                attention_mask=attention_mask,
            )
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
            raise RuntimeError(
                "GLM-5-Next MRoPE positions must have shape (3, sequence_length)."
            )

        # TokenizerManager currently consumes the processor result as a
        # mapping (membership checks and subscription), matching GLM4V's
        # established contract in this branch.
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.IM_TOKEN_ID,
            "im_start_id": self.IMAGE_START_TOKEN_ID,
            "im_end_id": self.IMAGE_END_TOKEN_ID,
            "video_token_id": self.VIDEO_TOKEN_ID,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
    "Glm5NextSGLangProcessor",
]
