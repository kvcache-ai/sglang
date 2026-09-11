# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sglang._vendor.compressed_tensors.compressors.base import BaseCompressor
from sglang._vendor.compressed_tensors.config import CompressionFormat
from sglang._vendor.compressed_tensors.quantization import QuantizationScheme
from sglang._vendor.compressed_tensors.utils import TensorStateDict


__all__ = ["DenseCompressor"]


@BaseCompressor.register(name=CompressionFormat.dense.value)
class DenseCompressor(BaseCompressor):
    """
    Identity compressor for dense models — both compress and decompress
    return the state dict unchanged.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        For dense models, this is a no-op.

        :param state_dict: local-name state dict (weight, ...)
        :param scheme: quantization scheme (unused)
        :return: unmodified state dict
        """
        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        For dense models, this is a no-op.

        :param state_dict: local-name state dict (weight, ...)
        :param scheme: quantization scheme (unused)
        :return: unmodified state dict
        """
        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Dense compressor matches when there's no weight quantization."""
        return True
