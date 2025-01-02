""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadataBuilder, AttentionType)
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import (
    CommonAttentionState, compute_slot_mapping,
    compute_slot_mapping_start_idx, is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

_PARTITION_SIZE = 512


class IpexAttnBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "IPEX"

    @staticmethod
    def get_impl_cls() -> Type["IpexAttnBackendImpl"]:
        return IpexAttnBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["IpexAttnMetadata"]:
        return IpexAttnMetadata

    @staticmethod
    def get_builder_cls() -> Type["IpexAttnMetadataBuilder"]:
        return IpexAttnMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        from vllm._ipex_ops import ipex_ops as ops
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        from vllm._ipex_ops import ipex_ops as ops
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class IpexAttnMetadata(FlashAttentionMetadata):
    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        raise NotImplementedError

class IpexAttnMetadataBuilder(
    AttentionMetadataBuilder[IpexAttnMetadata]):

    def __init__(self, input_builder: "ModelInputForXPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def _add_seq_group(
            self, inter_data: "ModelInputForXPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
            inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens, inter_data.seq_lens,
            inter_data.query_lens, inter_data.context_lens,
            inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                                  -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        num_seqs = len(seq_lens)
        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
        )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return IpexAttnMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
        )

class IpexAttnBackendImpl(AttentionImpl[IpexAttnMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "IPEX backend does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "IPEX backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IpexAttnBackendImpl")

    def ipex_attn_chunked_prefill(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: IpexAttnMetadata,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        kv_cache: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        scale: float,
        sliding_window: Optional[List[int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        print(attn_metadata)
        if attn_metadata is None:
            # Profiling run.
            return
        num_actual_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)
        print("!!!!!!kav")
        print(kv_cache)
        key_cache = None
        value_cache = None
        if kv_cache.numel() > 0:
            print("!!!!!!!kv_cache.numel() > 0")
            key_cache, value_cache = self.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            ipex_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
        if (kv_cache.numel() == 0 or attn_metadata.block_tables is None
                or attn_metadata.block_tables.numel() == 0):
            key_cache = key[:attn_metadata.num_prefill_tokens]
            value_cache = value[:attn_metadata.num_prefill_tokens]
        max_seq_len = max(attn_metadata.seq_lens)
        ipex_ops.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            None,
            attn_metadata.block_tables,
            alibi_slopes,
            attn_metadata.max_query_len,
            max_seq_len,
            0.0,
            scale,
            False,
            True,
            False,
            None,
        )


    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 1
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IpexAttnMetadata,  # type: ignore
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEX varlen_attention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # NOTE(woosuk): IPEXAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in IPEXAttention.")

        output = torch.empty_like(query)
        print("!!!!!!!key query")
        print(key)
        print(query)
        self.ipex_attn_chunked_prefill(
            output,
            query,
            key,
            value,
            attn_metadata,
            self.num_heads,
            self.head_size,
            self.num_kv_heads,
            kv_cache,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
            self.scale,
            self.sliding_window,
            self.alibi_slopes,
            self.logits_soft_cap,
        )
        return output.view(-1, self.num_heads * self.head_size)
    '''
    def ipex_attn_chunked_prefill(
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: IpexAttnMetadata,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        kv_cache: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        scale: float,
        sliding_window: Optional[List[int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if attn_metadata is None:
            # Profiling run.
            return
        num_actual_tokens = attn_metadata.num_actual_tokens

        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        # Reshape the input keys and values and store them in the cache.
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        ipex_ops.reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        ipex_ops.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            None,
            attn_metadata.block_table,
            alibi_slopes,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            0.0,
            scale,
            False,
            True,
            False,
            None,
        )
        '''
