"""Focused tests for the DeepSeek V4 FlashInfer decode path on SM120."""

import unittest

import torch

from sglang.srt.layers.attention import flash_mla_sm120 as sm120
from sglang.srt.layers.attention.debug_flash_mla_adapter import (
    _v4_triton_decode_dispatch,
)

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _build_kv_cache(
    num_pages: int, page_size: int, device: torch.device, seed: int
) -> torch.Tensor:
    """Build valid FP8/BF16/UE8M0 bytes in DSV4's footer layout."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nope = (
        torch.randn(num_pages, page_size, 448, generator=generator, dtype=torch.float32)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(device)
    )
    rope = (
        torch.randn(num_pages, page_size, 64, generator=generator, dtype=torch.float32)
        .clamp(-2, 2)
        .to(torch.bfloat16)
        .view(torch.uint8)
        .to(device)
    )

    token_data = torch.empty(
        num_pages,
        page_size,
        sm120._NOPE_ROPE_STRIDE,
        dtype=torch.uint8,
        device=device,
    )
    token_data[:, :, :448] = nope
    token_data[:, :, 448:] = rope
    scales = torch.zeros(
        num_pages,
        page_size,
        sm120._SCALE_STRIDE,
        dtype=torch.uint8,
        device=device,
    )
    scales[:, :, :7] = 127  # UE8M0 encoding for scale 1.0.

    flat = torch.empty(
        num_pages,
        page_size * sm120._BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=device,
    )
    data_bytes = page_size * sm120._NOPE_ROPE_STRIDE
    flat[:, :data_bytes] = token_data.reshape(num_pages, data_bytes)
    flat[:, data_bytes:] = scales.reshape(num_pages, page_size * sm120._SCALE_STRIDE)
    return flat.view(num_pages, page_size, 1, sm120._BYTES_PER_TOKEN)


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestDsv4TouchedPageSplit(unittest.TestCase):
    def test_only_touched_pages_are_rewritten(self):
        device = torch.device("cuda")
        num_pages = 3
        src_page_size = 128
        ratio = src_page_size // sm120._PBS_DST
        sentinel = 0xA5
        cache = _build_kv_cache(num_pages, src_page_size, device, seed=17)
        src_stride = cache.stride(0)
        src = torch.as_strided(cache, (num_pages, src_stride), (src_stride, 1))

        key = sm120._device_key(cache.device)
        old_split = sm120._SPLIT_BUFFERS.get(key)
        old_mask = sm120._MASK_BUFFERS.get(key)

        def restore():
            if old_split is None:
                sm120._SPLIT_BUFFERS.pop(key, None)
            else:
                sm120._SPLIT_BUFFERS[key] = old_split
            if old_mask is None:
                sm120._MASK_BUFFERS.pop(key, None)
            else:
                sm120._MASK_BUFFERS[key] = old_mask

        self.addCleanup(restore)
        dst = torch.full(
            (num_pages * ratio, sm120._BYTES_PER_DST_PAGE_PADDED),
            sentinel,
            dtype=torch.uint8,
            device=device,
        )
        sm120._SPLIT_BUFFERS[key] = dst
        indices = torch.tensor(
            [0, 5, 2 * src_page_size + 3, -1],
            dtype=torch.int32,
            device=device,
        )

        with torch.inference_mode():
            sm120._split_kv_pages_to_64(cache, src_page_size, indices)
        torch.cuda.synchronize()
        self.assertFalse(sm120._MASK_BUFFERS[key].is_inference())

        dst.fill_(sentinel)
        sm120._MASK_BUFFERS[key].fill_(-7)
        out = sm120._split_kv_pages_to_64(cache, src_page_size, indices)
        torch.cuda.synchronize()
        self.assertEqual(
            out.shape,
            (num_pages * ratio, sm120._PBS_DST, 1, sm120._BYTES_PER_TOKEN),
        )
        self.assertEqual(sm120._MASK_BUFFERS[key].tolist(), [1, 0, 1])

        data_size = sm120._PBS_DST * sm120._NOPE_ROPE_STRIDE
        scale_size = sm120._PBS_DST * sm120._SCALE_STRIDE
        src_scale_offset = src_page_size * sm120._NOPE_ROPE_STRIDE
        for page in (0, 2):
            for subpage in range(ratio):
                dst_page = dst[page * ratio + subpage]
                torch.testing.assert_close(
                    dst_page[:data_size],
                    src[
                        page,
                        subpage * data_size : (subpage + 1) * data_size,
                    ],
                    atol=0,
                    rtol=0,
                )
                scale_offset = src_scale_offset + subpage * scale_size
                torch.testing.assert_close(
                    dst_page[data_size : data_size + scale_size],
                    src[page, scale_offset : scale_offset + scale_size],
                    atol=0,
                    rtol=0,
                )
                self.assertTrue(
                    bool((dst_page[sm120._BYTES_PER_DST_PAGE :] == sentinel).all())
                )
        self.assertTrue(bool((dst[ratio : 2 * ratio] == sentinel).all()))


@unittest.skipUnless(_IS_SM120, "SM120 (compute capability 12.0) required")
class TestDsv4FlashInferDecode(unittest.TestCase):
    def test_flashinfer_matches_existing_triton_reference(self):
        if not sm120.is_flashinfer_dsv4_available():
            self.skipTest("FlashInfer SM120 DSV4 sparse MLA API unavailable")

        device = torch.device("cuda")
        generator = torch.Generator(device="cpu").manual_seed(29)
        # This tuple hits FlashInfer's dedicated DSV4 decode dispatch rather
        # than its generic sparse-MLA fallback.
        batch_size, num_heads, topk = 1, 128, 128
        num_pages, page_size = 4, 128
        cache = _build_kv_cache(num_pages, page_size, device, seed=23)
        q = (
            torch.randn(
                batch_size,
                1,
                num_heads,
                512,
                generator=generator,
                dtype=torch.float32,
            )
            .clamp(-1.5, 1.5)
            .to(torch.bfloat16)
            .to(device)
        )
        indices = torch.randperm(
            num_pages * page_size, generator=generator, dtype=torch.int64
        )[:topk].to(device=device, dtype=torch.int32)
        indices = indices.view(batch_size, 1, topk)
        lengths = torch.full((batch_size,), topk, dtype=torch.int32, device=device)
        sink = torch.full((num_heads,), -4.0, dtype=torch.float32, device=device)
        kwargs = dict(
            q=q,
            k_cache=cache,
            head_dim_v=512,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=lengths,
            attn_sink=sink,
        )

        expected, _ = _v4_triton_decode_dispatch(**kwargs)
        actual, _ = sm120.flash_mla_with_kvcache_sm120(**kwargs)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=5e-2, rtol=5e-2
        )


if __name__ == "__main__":
    unittest.main()
