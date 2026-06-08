# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 MXFP8 quantization for diffusion transformers on Ascend NPU."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import CopyNumelCounter, _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


class DiffusionMXFP8Config(QuantizationConfig):
    """W8A8 MXFP8 config for diffusion LinearBase layers.

    ``is_checkpoint_mxfp8_serialized=True`` selects the offline path for
    checkpoints that already contain MXFP8 weights and ``weight_scale`` tensors.
    The default path loads BF16/FP16 weights and quantizes them at load time.
    """

    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp8_serialized = is_checkpoint_mxfp8_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP8Config:
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_mxfp8_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_mxfp8_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        if not current_omni_platform.is_npu():
            raise NotImplementedError("MXFP8 diffusion quantization is currently only supported on Ascend NPU.")

        if self.is_checkpoint_mxfp8_serialized:
            return NPUMxfp8LinearMethod(self)
        return NPUMxfp8OnlineLinearMethod(self)


class _LazyWeightMixin:
    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            param = layer.weight
            counter = CopyNumelCounter()
            with counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            layer._loaded_numel += counter.copied_numel

            if layer._loaded_numel == layer.weight.numel():
                self.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True
            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)


class MXFPLinearMethodBase(LinearMethodBase, ABC):
    @abstractmethod
    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _apply_inner(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        x_q, x_scale = self._quantize_activation(x)
        return self._quant_matmul(x_q, x_scale, layer, bias, ori_dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        output = self._apply_inner(layer, x, bias, ori_dtype)
        return output.reshape(*ori_shape[:-1], -1)


class NPUMxfp8LinearMethod(MXFPLinearMethodBase):
    """NPU offline MXFP8 linear method for serialized checkpoints."""

    def __init__(self, quant_config: DiffusionMXFP8Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        num_groups = (input_size_per_partition + 31) // 32
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups, dtype=torch.uint8),
                input_dim=None,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        weight = layer.weight
        if weight.dtype != torch_npu.float8_e4m3fn:
            weight = torch_npu.npu_dtype_cast(weight.npu(), torch_npu.float8_e4m3fn)
        weight = weight.transpose(0, 1).contiguous()

        scale = layer.weight_scale.data
        if scale.dtype not in (torch.uint8, torch_npu.float8_e8m0fnu):
            scale = scale.to(torch_npu.float8_e8m0fnu)
        out_features, num_groups = scale.shape
        if num_groups % 2 == 1:
            scale = torch.cat([scale, torch.zeros(out_features, 1, dtype=scale.dtype, device=scale.device)], dim=1)
            num_groups += 1
        scale = scale.reshape(out_features, num_groups // 2, 2).transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", scale)
        layer._already_called_process_weights_after_loading = True

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        return torch_npu.npu_quant_matmul(
            x_q,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )


class NPUMxfp8OnlineLinearMethod(_LazyWeightMixin, NPUMxfp8LinearMethod):
    """NPU MXFP8 online method: load BF16/FP16 weights, then quantize."""

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import torch_npu

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        weight_fp8, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(
            layer.weight, dst_type=torch_npu.float8_e4m3fn
        )
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2).transpose(0, 1).contiguous()
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", weight_fp8)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True
