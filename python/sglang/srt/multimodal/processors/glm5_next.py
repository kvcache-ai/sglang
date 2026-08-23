"""Strict image/video request adapter for GLM-5-Next.

The checkpoint processor lives in transformers-kt.  SGLang owns only the
request boundary, scheduler metadata, and the invariants needed by its vision
embedding path; patchification and temporal sampling have one source of truth.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from contextlib import contextmanager
from urllib.parse import unquote, urlparse

import requests
import torch
from transformers.models.glm5_next.image_processing_glm5_next import (
    Glm5NextImageProcessor,
)
from transformers.models.glm5_next.processing_glm5_next import Glm5NextProcessor
from transformers.models.glm5_next.video_processing_glm5_next import (
    Glm5NextVideoProcessor,
)

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor


GLM5_NEXT_MIN_PIXELS = 12_544
GLM5_NEXT_DEFAULT_MAX_PIXELS = 8000 * 14**2 * 2**2
GLM5_NEXT_CHECKPOINT_MAX_PIXELS = GLM5_NEXT_DEFAULT_MAX_PIXELS
GLM5_NEXT_PATCH_EXPAND_FACTOR = 1


class Glm5NextSGLangProcessor(Glm4vImageProcessor):
    """Strict text+images or text+one-video adapter for GLM-5-Next."""

    models = [Glm5NextForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = GLM5_NEXT_MIN_PIXELS
        self.MAX_PIXELS = self._resolve_max_pixels(server_args.mm_process_config)

        image_processor = getattr(_processor, "image_processor", None)
        if not isinstance(image_processor, Glm5NextImageProcessor):
            raise TypeError(
                "GLM-5-Next requires the built-in Glm5NextImageProcessor; "
                f"got {type(image_processor).__name__}."
            )
        if image_processor.patch_expand_factor != GLM5_NEXT_PATCH_EXPAND_FACTOR:
            raise ValueError(
                "GLM-5-Next requires image patch_expand_factor=1; "
                f"got {image_processor.patch_expand_factor!r}."
            )
        if isinstance(image_processor.size, dict):
            image_processor.size["shortest_edge"] = self.MIN_PIXELS
            image_processor.size["longest_edge"] = self.MAX_PIXELS
        else:
            image_processor.size.shortest_edge = self.MIN_PIXELS
            image_processor.size.longest_edge = self.MAX_PIXELS

        video_processor = getattr(_processor, "video_processor", None)
        if not isinstance(video_processor, Glm5NextVideoProcessor):
            raise TypeError(
                "GLM-5-Next requires Glm5NextVideoProcessor; "
                f"got {type(video_processor).__name__}."
            )
        if video_processor.patch_expand_factor != GLM5_NEXT_PATCH_EXPAND_FACTOR:
            raise ValueError(
                "GLM-5-Next requires video patch_expand_factor=1; "
                f"got {video_processor.patch_expand_factor!r}."
            )
        if video_processor.fps != 2:
            raise ValueError(
                f"GLM-5-Next requires the checkpoint video fps=2; got {video_processor.fps!r}."
            )

        # GLM expands video frames to the image placeholder id.  Keep the
        # source token distinct but advertise the post-tokenization id used by
        # the scheduler and embedding scatter.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    @staticmethod
    def _resolve_max_pixels(mm_process_config) -> int:
        config = {} if mm_process_config is None else mm_process_config
        if not isinstance(config, dict):
            raise ValueError("GLM-5-Next --mm-process-config must be a JSON object.")
        unknown_root = set(config) - {"image"}
        if unknown_root:
            raise ValueError(
                "GLM-5-Next accepts only the image processor override; "
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

    def _count_placeholders(self, input_text) -> tuple[int, int]:
        if isinstance(input_text, str):
            return (
                input_text.count(self.IMAGE_TOKEN),
                input_text.count(self.VIDEO_TOKEN),
            )
        if isinstance(input_text, (list, tuple)):
            return (
                input_text.count(self.IM_TOKEN_ID),
                input_text.count(self.VIDEO_TOKEN_ID),
            )
        raise TypeError("GLM-5-Next multimodal prompt must be text or token ids.")

    @staticmethod
    @contextmanager
    def _materialize_video(video):
        """Yield one seekable local file for transformers/PyAV and clean it up."""

        max_bytes = int(
            os.environ.get("SGLANG_GLM5_NEXT_MAX_VIDEO_BYTES", 2 * 1024**3)
        )
        if max_bytes <= 0:
            raise ValueError("SGLANG_GLM5_NEXT_MAX_VIDEO_BYTES must be positive.")
        temporary_path = None
        try:
            if isinstance(video, str):
                parsed = urlparse(video)
                if parsed.scheme == "file":
                    local_path = unquote(parsed.path)
                    if not os.path.isfile(local_path):
                        raise ValueError(f"GLM-5-Next video file not found: {local_path}")
                    yield local_path
                    return
                if parsed.scheme in ("", None) and os.path.isfile(video):
                    yield video
                    return

            payload = None
            if isinstance(video, bytes):
                payload = video
            elif isinstance(video, str) and video.startswith(("http://", "https://")):
                timeout = int(os.environ.get("REQUEST_TIMEOUT", "10"))
                response = requests.get(video, stream=True, timeout=timeout)
                try:
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > max_bytes:
                        raise ValueError(
                            "GLM-5-Next video exceeds the configured byte limit: "
                            f"content_length={content_length}, limit={max_bytes}."
                        )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ) as temporary:
                        temporary_path = temporary.name
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise ValueError(
                                    "GLM-5-Next video exceeds the configured byte "
                                    f"limit of {max_bytes} bytes."
                                )
                            temporary.write(chunk)
                finally:
                    response.close()
                if downloaded == 0:
                    raise ValueError("GLM-5-Next downloaded an empty video.")
                yield temporary_path
                return
            elif isinstance(video, str):
                encoded = video.split(",", 1)[1] if video.startswith("data:") else video
                try:
                    payload = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as error:
                    raise ValueError(
                        "GLM-5-Next video must be a local path, HTTP(S) URL, "
                        "bytes, data URL, or valid base64."
                    ) from error
            else:
                raise ValueError(
                    f"Unsupported GLM-5-Next video input type: {type(video).__name__}."
                )

            if not payload:
                raise ValueError("GLM-5-Next video payload is empty.")
            if len(payload) > max_bytes:
                raise ValueError(
                    "GLM-5-Next video exceeds the configured byte limit: "
                    f"size={len(payload)}, limit={max_bytes}."
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temporary:
                temporary_path = temporary.name
                temporary.write(payload)
            yield temporary_path
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

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
            raise ValueError("GLM-5-Next does not support audio input.")
        video_data = getattr(request_obj, "video_data", None) or []
        image_data = image_data or []
        if image_data and video_data:
            raise ValueError(
                "GLM-5-Next does not allow image and video content in one request."
            )
        if image_data:
            return self._process_image_data(image_data, input_text)
        if video_data:
            return self._process_video_data(video_data, input_text)
        raise ValueError("GLM-5-Next multimodal request contains no media data.")

    def _process_image_data(self, image_data, input_text):
        if not isinstance(image_data, list) or not image_data:
            count = len(image_data) if isinstance(image_data, list) else 0
            raise ValueError(
                "GLM-5-Next requires at least one image per image request; "
                f"got {count}."
            )
        image_count = len(image_data)
        if image_count > 8:
            raise ValueError(
                f"GLM-5-Next accepts at most 8 images per request; got {image_count}."
            )
        if any(isinstance(image, dict) for image in image_data):
            raise ValueError(
                "GLM-5-Next does not accept processor_output or "
                "precomputed_embedding image inputs."
            )
        image_placeholders, video_placeholders = self._count_placeholders(input_text)
        if video_placeholders:
            raise ValueError(
                "GLM-5-Next image request unexpectedly contains a video placeholder."
            )
        if image_placeholders != image_count:
            raise ValueError(
                "GLM-5-Next image count/placeholder count mismatch: "
                f"images={image_count}, placeholders={image_placeholders}."
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
            "video_token_id": self.IM_TOKEN_ID,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
            "glm5_next_force_hybrid_prefill": True,
        }

    def _process_video_data(self, video_data, input_text):
        if not isinstance(video_data, list) or len(video_data) != 1:
            count = len(video_data) if isinstance(video_data, list) else 0
            raise ValueError(
                f"GLM-5-Next accepts exactly one video per video request; got {count}."
            )
        if isinstance(video_data[0], dict):
            raise ValueError(
                "GLM-5-Next does not accept processor_output or "
                "precomputed_embedding video inputs."
            )

        image_placeholders, video_placeholders = self._count_placeholders(input_text)
        if image_placeholders:
            raise ValueError(
                "GLM-5-Next video request unexpectedly contains an image placeholder."
            )
        if video_placeholders != 1:
            raise ValueError(
                "GLM-5-Next video count/placeholder count mismatch: "
                f"videos=1, placeholders={video_placeholders}."
            )

        if isinstance(input_text, (list, tuple)):
            input_text = self._processor.tokenizer.decode(input_text)
        # Do not send videos through SGLang's Decord loader and then decode a
        # second time.  The transformers-kt video processor owns PyAV sampling
        # and returns the metadata used to build timestamp tokens.
        with self._materialize_video(video_data[0]) as video_path:
            base_output = BaseMultiModalProcessorOutput(
                input_text=input_text,
                images=[],
                videos=[video_path],
                audios=[],
            )
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens, return_metadata=True
            )
        video_grid_thw = getattr(ret, "video_grid_thw", None)
        if video_grid_thw is None and isinstance(ret, dict):
            video_grid_thw = ret.get("video_grid_thw")
        if video_grid_thw is None or tuple(video_grid_thw.shape) != (1, 3):
            actual_shape = (
                None if video_grid_thw is None else tuple(video_grid_thw.shape)
            )
            raise RuntimeError(
                "GLM-5-Next processor returned an invalid video_grid_thw shape: "
                f"expected=(1, 3), got={actual_shape}."
            )
        grid_t, grid_h, grid_w = (int(value) for value in video_grid_thw[0].tolist())
        if min(grid_t, grid_h, grid_w) <= 0:
            raise RuntimeError(
                "GLM-5-Next video grid dimensions must be positive; "
                f"got {(grid_t, grid_h, grid_w)}."
            )
        merge_size = int(self.spatial_merge_size)
        if grid_h % merge_size or grid_w % merge_size:
            raise RuntimeError(
                "GLM-5-Next video grid is not divisible by the spatial merge "
                f"size: grid={(grid_t, grid_h, grid_w)}, merge_size={merge_size}."
            )
        if len(mm_items) != 1 or not mm_items[0].is_video():
            raise RuntimeError("GLM-5-Next processor must return one video item.")

        expected_feature_rows = grid_t * grid_h * grid_w
        feature = mm_items[0].feature
        feature_shape = getattr(feature, "shape", None)
        if feature_shape is None:
            feature_shape = getattr(getattr(feature, "info_data", None), "shape", None)
        if (
            feature_shape is None
            or len(feature_shape) != 2
            or int(feature_shape[0]) != expected_feature_rows
        ):
            raise RuntimeError(
                "GLM-5-Next video feature/grid mismatch: "
                f"expected_rows={expected_feature_rows}, "
                f"feature_shape={None if feature_shape is None else tuple(feature_shape)}."
            )

        tokens_per_frame = grid_h * grid_w // (merge_size**2)
        expected_video_tokens = grid_t * tokens_per_frame
        video_token_positions = (
            (input_ids == self.IM_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        )
        if len(video_token_positions) != expected_video_tokens:
            raise RuntimeError(
                "GLM-5-Next video placeholder/feature mismatch: "
                f"expected={expected_video_tokens}, got={len(video_token_positions)}."
            )
        offsets = mm_items[0].offsets or []
        if len(offsets) != grid_t:
            raise RuntimeError(
                "GLM-5-Next video must expose one image-token block per temporal "
                f"grid row: expected={grid_t}, got={len(offsets)}."
            )
        covered_positions = []
        for frame_index, (start, end) in enumerate(offsets):
            start, end = int(start), int(end)
            if end - start + 1 != tokens_per_frame:
                raise RuntimeError(
                    "GLM-5-Next video frame token block length mismatch at frame "
                    f"{frame_index}: expected={tokens_per_frame}, got={end - start + 1}."
                )
            covered_positions.extend(range(start, end + 1))
        if covered_positions != video_token_positions:
            raise RuntimeError(
                "GLM-5-Next video offsets do not cover exactly the expanded frame tokens."
            )

        video_start_positions = (
            (input_ids == self.VIDEO_START_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        )
        video_end_positions = (
            (input_ids == self.VIDEO_END_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        )
        if (
            len(video_start_positions) != 1
            or len(video_end_positions) != 1
            or video_start_positions[0] >= video_end_positions[0]
        ):
            raise RuntimeError(
                "GLM-5-Next video prompt must contain one ordered begin/end boundary."
            )
        video_start, video_end = video_start_positions[0], video_end_positions[0]
        image_starts = (
            (input_ids[video_start + 1 : video_end] == self.IMAGE_START_TOKEN_ID)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        image_ends = (
            (input_ids[video_start + 1 : video_end] == self.IMAGE_END_TOKEN_ID)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        if len(image_starts) != grid_t or len(image_ends) != grid_t:
            raise RuntimeError(
                "GLM-5-Next video frame boundaries/grid mismatch: "
                f"grid_t={grid_t}, starts={len(image_starts)}, ends={len(image_ends)}."
            )
        if any(start >= end for start, end in zip(image_starts, image_ends)):
            raise RuntimeError("GLM-5-Next video frame boundaries are not ordered.")

        video_metadata = getattr(ret, "video_metadata", None)
        if video_metadata is None and isinstance(ret, dict):
            video_metadata = ret.get("video_metadata")
        # BatchFeature normalizes non-tensor metadata to a tuple on the
        # transformers-kt 5.6 release line, while newer releases preserve a
        # list. Both representations carry the same ordered request metadata.
        if not isinstance(video_metadata, (list, tuple)) or len(video_metadata) != 1:
            raise RuntimeError(
                "GLM-5-Next video processor must return exactly one metadata record."
            )
        metadata = video_metadata[0]
        timestamps = list(metadata.timestamps[::2])[:grid_t]
        while len(timestamps) < grid_t:
            timestamps.append(timestamps[-1] if timestamps else 0.0)
        if len(timestamps) != grid_t or any(
            not torch.isfinite(torch.tensor(timestamp)) for timestamp in timestamps
        ):
            raise RuntimeError("GLM-5-Next video timestamps are invalid.")
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise RuntimeError("GLM-5-Next video timestamps must be monotonic.")

        expected_frame_text = "".join(
            self._processor.replace_frame_token_id(timestamp)
            for timestamp in timestamps
        )
        expected_frame_ids = self._processor.tokenizer.encode(
            expected_frame_text, add_special_tokens=False
        )
        actual_frame_ids = input_ids[video_start + 1 : video_end].tolist()
        if actual_frame_ids != expected_frame_ids:
            raise RuntimeError(
                "GLM-5-Next timestamped video token expansion changed unexpectedly."
            )

        attention_mask = getattr(ret, "attention_mask", None)
        if attention_mask is None and isinstance(ret, dict):
            attention_mask = ret.get("attention_mask")
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        mrope_positions = mrope_positions.squeeze(1)
        if mrope_positions.ndim != 2 or tuple(mrope_positions.shape) != (
            3,
            input_ids.numel(),
        ):
            raise RuntimeError(
                "GLM-5-Next video MRoPE positions must have shape (3, sequence_length)."
            )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.IM_TOKEN_ID,
            "im_start_id": self.IMAGE_START_TOKEN_ID,
            "im_end_id": self.IMAGE_END_TOKEN_ID,
            "video_token_id": self.IM_TOKEN_ID,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
            "glm5_next_force_hybrid_prefill": True,
        }


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
    "Glm5NextSGLangProcessor",
    "Glm5NextVideoProcessor",
]
