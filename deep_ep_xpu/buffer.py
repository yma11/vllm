import warnings

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import deep_ep_cpp
# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, EventHandle
from .utils import EventOverlap


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine using IPC)

    Attributes:
        num_eus: the EUs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_ipc_bytes: the buffer size for intranode IPC communication.
        num_rdma_bytes: the buffer size for internode RDMA communication (not yet supported on XPU).
        runtime: the C++ runtime.
    """

    num_eus: int = 20

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 num_ipc_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_ipc_bytes: the buffer size for intranode IPC communication.
            num_rdma_bytes: the buffer size for internode RDMA communication (not yet supported on XPU).
            low_latency_mode: whether to enable low-latency mode (not yet supported on XPU).
            enable_shrink: whether to enable shrink mode.
            explicitly_destroy: If True, you must explicitly call `destroy()` to release resources.
            comm: the `mpi4py.MPI.Comm` communicator to use when group is absent.
        """

        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")

        self.num_ipc_bytes = num_ipc_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink

        self.runtime = deep_ep_cpp.Buffer(
            self.rank,
            self.group_size,
            num_ipc_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
            enable_shrink
        )

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles via Ring AllGather (Unix socket + SCM_RIGHTS fd passing)
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        if comm is not None:
            barrier_func = comm.Barrier
        else:
            barrier_func = lambda: dist.barrier(group=self.group)
        ipc_handles = self.runtime.all_gather_handle(local_ipc_handle, barrier_func)

        # InterNode is not yet supported on XPU
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1:
            warnings.warn("InterNode communication is not yet supported on XPU. "
                          "Only intranode IPC communication is available.")

        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """
        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def set_num_eus(new_num_eus: int) -> None:
        """
        Set the number of EUs to use in high-throughput kernels.

        Arguments:
            new_num_eus: the new number to be set.
        """

        assert new_num_eus % 2 == 0, 'The EU count must be even'
        Buffer.num_eus = new_num_eus

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a XPU event on the current stream, i.e. `torch.xpu.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_queue()
        return torch.xpu.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: These config values are ported from the NVIDIA implementation and need to be
        # re-tuned for Intel XPU (PVC) architecture.
        config_map = {
            2: Config(Buffer.num_eus, 24, 256, 0, 0),
            4: Config(Buffer.num_eus, 6, 256, 0, 0),
            8: Config(Buffer.num_eus, 6, 256, 0, 0),
            16: Config(Buffer.num_eus, 36, 288, 0, 0),
            24: Config(Buffer.num_eus, 32, 288, 0, 0),
            32: Config(Buffer.num_eus, 32, 288, 0, 0),
            48: Config(Buffer.num_eus, 32, 288, 0, 0),
            64: Config(Buffer.num_eus, 32, 288, 0, 0),
            96: Config(Buffer.num_eus, 20, 480, 0, 0),
            128: Config(Buffer.num_eus, 20, 560, 0, 0),
            144: Config(Buffer.num_eus, 32, 720, 0, 0),
            160: Config(Buffer.num_eus, 28, 720, 0, 0),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: These config values are ported from the NVIDIA implementation and need to be
        # re-tuned for Intel XPU.
        config_map = {
            2: Config(Buffer.num_eus, 10, 256, 0, 0),
            4: Config(Buffer.num_eus, 9, 256, 0, 0),
            8: Config(Buffer.num_eus, 4, 256, 0, 0),
            16: Config(Buffer.num_eus, 4, 288, 0, 0),
            24: Config(Buffer.num_eus, 1, 288, 0, 0),
            32: Config(Buffer.num_eus, 1, 288, 0, 0),
            48: Config(Buffer.num_eus, 1, 288, 0, 0),
            64: Config(Buffer.num_eus, 1, 288, 0, 0),
            96: Config(Buffer.num_eus, 1, 480, 0, 0),
            128: Config(Buffer.num_eus, 1, 560, 0, 0),
            144: Config(Buffer.num_eus, 2, 720, 0, 0),
            160: Config(Buffer.num_eus, 2, 720, 0, 0),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]


    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, 
                 num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, 
                 num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, 
                 topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1, 
                 num_worst_tokens: int = 0,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, 
                 async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via IPC.
        Internode kernels require the ranks in a node should be visible via IPC, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be XPU-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config
        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,
                                           num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, num_worst_tokens, config,
                                           previous_event, async_finish, allocate_on_comm_stream)

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, 
             num_recv_tokens_per_expert_list, rank_prefix_matrix, 
             channel_prefix_matrix, recv_channel_prefix_matrix, 
             recv_src_idx, send_head, event) = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via IPC.
        Internode kernels require the ranks in a node should be visible via IPC, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)

        # NOTES: the second `_` is for the sending side, so we should use the third one
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
                                                                          channel_prefix_matrix, send_head, config,
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           num_worst_tokens: int = 0, config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        raise NotImplementedError("Internode is not supported in deep_ep_xpu yet.")


    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        raise NotImplementedError("Internode is not supported in deep_ep_xpu yet.")

