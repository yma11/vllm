#include "deep_ep.hpp"

#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <chrono>
#include <iostream>
#include <memory>

#include "sycl/api.hpp"
#include "sycl/layout.hpp"
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include "sycl/configs.h"
#include "sycl/utils.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <pwd.h>
#include <cstring>


namespace shared_memory {

SharedMemoryAllocator::SharedMemoryAllocator(sycl::queue& queue) {
    ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
    ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
}

void SharedMemoryAllocator::init_from_queue(sycl::queue& queue) {
    ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
    ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.flags = 0;
    alloc_desc.ordinal = 0;

    ze_result_t result = zeMemAllocDevice(ze_context, &alloc_desc, size_raw, 64, ze_device, ptr);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to allocate XPU device memory");
}

void SharedMemoryAllocator::free(void* ptr) {
    ze_result_t result = zeMemFree(ze_context, ptr);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to free XPU device memory");
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    void* base_addr;
    size_t base_size;
    ze_result_t result = zeMemGetAddressRange(ze_context, ptr, &base_addr, &base_size);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to get XPU memory address range");

    mem_handle->size = base_size;
    result = zeMemGetIpcHandle(ze_context, base_addr, &mem_handle->inner.ze_ipc_mem_handle);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to get XPU IPC handle");
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
    ze_result_t result = zeMemOpenIpcHandle(
        ze_context, ze_device,
        mem_handle->inner.ze_ipc_mem_handle,
        ZE_IPC_MEMORY_FLAG_BIAS_CACHED,
        ptr);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to open XPU IPC handle");
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    ze_result_t result = zeMemCloseIpcHandle(ze_context, ptr);
    EP_HOST_ASSERT(result == ZE_RESULT_SUCCESS && "Failed to close XPU IPC handle");
}

}  // namespace shared_memory


namespace deep_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_ipc_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink)
    : rank(rank),
      num_ranks(num_ranks),
      num_ipc_bytes(num_ipc_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      shared_memory_allocator() {

    // Select GPU device by rank
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    EP_HOST_ASSERT(rank < static_cast<int>(devices.size()) &&
                   "Rank exceeds number of available GPU devices");

    sycl::device target_device = devices[rank];
    comm_stream = sycl::queue(target_device, sycl::property::queue::in_order{});
    device_id = rank;

    // Initialize Level Zero allocator with correct device
    shared_memory_allocator.init_from_queue(comm_stream);

    // Calculate rank positions
    rdma_rank = rank / NUM_MAX_IPC_PEERS;
    ipc_rank = rank % NUM_MAX_IPC_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_IPC_PEERS);
    num_ipc_ranks = std::min(num_ranks, NUM_MAX_IPC_PEERS);

    // Get compute unit count
    num_device_eus = comm_stream.get_device().get_info<sycl::info::device::max_compute_units>();

    // Metadata layout
    int64_t barrier_signal_bytes = NUM_MAX_IPC_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_IPC_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_IPC_PEERS * sizeof(int*);

    // Alignment checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_ipc_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (num_ipc_bytes <= std::numeric_limits<int>::max() || num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                   (low_latency_mode || num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_ipc_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank && rank < num_ranks &&
                   (num_ranks <= NUM_MAX_IPC_PEERS * NUM_MAX_RDMA_PEERS || low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_IPC_PEERS || num_ranks % NUM_MAX_IPC_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_IPC_PEERS || low_latency_mode);

    // Per-channel byte count check
    EP_HOST_ASSERT(ceil_div<int64_t>(num_ipc_bytes, num_device_eus / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_eus / 2) < std::numeric_limits<int>::max());

    // Allocate intranode IPC buffer
    if (num_ipc_bytes > 0) {
        shared_memory_allocator.malloc(&buffer_ptrs[ipc_rank],
                                       num_ipc_bytes +
                                       barrier_signal_bytes +
                                       buffer_ptr_bytes +
                                       barrier_signal_ptr_bytes);

        shared_memory_allocator.get_mem_handle(&ipc_handles[ipc_rank], buffer_ptrs[ipc_rank]);

        buffer_ptrs_gpu = reinterpret_cast<void**>(
            static_cast<uint8_t*>(buffer_ptrs[ipc_rank]) + num_ipc_bytes + barrier_signal_bytes);

        barrier_signal_ptrs[ipc_rank] = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs[ipc_rank]) + num_ipc_bytes);

        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
            static_cast<uint8_t*>(buffer_ptrs[ipc_rank]) + num_ipc_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        comm_stream.memset(barrier_signal_ptrs[ipc_rank], 0, barrier_signal_bytes).wait();
    }

    // Allocate workspace
    workspace = sycl::malloc_device(NUM_WORKSPACE_BYTES, comm_stream);
    comm_stream.memset(workspace, 0, NUM_WORKSPACE_BYTES).wait();

    // MoE counters (pinned host memory for host-device access)
    moe_recv_counter = const_cast<volatile int*>(
        static_cast<int*>(sycl::malloc_host(sizeof(int64_t), comm_stream)));
    moe_recv_counter_mapped = const_cast<int*>(moe_recv_counter);
    *moe_recv_counter = -1;

    moe_recv_expert_counter = const_cast<volatile int*>(
        static_cast<int*>(sycl::malloc_host(sizeof(int) * NUM_MAX_LOCAL_EXPERTS, comm_stream)));
    moe_recv_expert_counter_mapped = const_cast<int*>(moe_recv_expert_counter);
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    if (num_rdma_ranks > 1) {
        throw std::runtime_error("RDMA (internode) communication is not yet supported on XPU");
    }
}

Buffer::~Buffer() noexcept(false) {
    if (!explicitly_destroy) {
        destroy();
    } else if (!destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

void Buffer::destroy() {
    EP_HOST_ASSERT(!destroyed);

    if (num_ipc_bytes > 0) {
        // Synchronize before cleanup
        comm_stream.wait();

        // Close remote IPC handles
        if (is_available()) {
            for (int i = 0; i < num_ipc_ranks; ++i)
                if (i != ipc_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local IPC buffer
        shared_memory_allocator.free(buffer_ptrs[ipc_rank]);
    }

    // Free workspace
    sycl::free(workspace, comm_stream);

    // Free MoE counters
    sycl::free(const_cast<int*>(moe_recv_counter), comm_stream);
    sycl::free(const_cast<int*>(moe_recv_expert_counter), comm_stream);
    if (moe_recv_rdma_counter)
        sycl::free(const_cast<int*>(moe_recv_rdma_counter), comm_stream);

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(!is_available());

    // Sync IPC handles
    if (num_ipc_bytes > 0) {
        EP_HOST_ASSERT(static_cast<int>(device_ids.size()) == num_ranks);
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());

        for (int i = 0, offset = rdma_rank * num_ipc_ranks; i < num_ipc_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(
                    static_cast<uint8_t*>(buffer_ptrs[i]) + num_ipc_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
            }
        }

        // Copy buffer and barrier signal pointers to GPU
        comm_stream.memcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_IPC_PEERS);
        comm_stream.memcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_IPC_PEERS);
        comm_stream.wait();
    }

    available = true;
}


bool Buffer::is_available() const { return available; }

bool Buffer::is_internode_available() const {
    return false;
}

int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks; }

int Buffer::get_rdma_rank() const { return rdma_rank; }

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? ipc_rank : 0;
}

int Buffer::get_local_device_id() const { return device_id; }

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[ipc_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

sycl::queue Buffer::get_comm_queue() const { return comm_stream; }

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[ipc_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_ipc_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes,
                            torch::TensorOptions().dtype(casted_dtype).device(at::kXPU));
}


namespace {

struct exchange_fd {
    char obscure[CMSG_LEN(sizeof(int)) - sizeof(int)];
    int fd;

    exchange_fd(int cmsg_level, int cmsg_type, int fd) : fd(fd) {
        auto* cmsg = reinterpret_cast<cmsghdr*>(obscure);
        cmsg->cmsg_len = sizeof(exchange_fd);
        cmsg->cmsg_level = cmsg_level;
        cmsg->cmsg_type = cmsg_type;
    }

    exchange_fd() : fd(-1) {
        memset(obscure, 0, sizeof(obscure));
    }
};

struct IpcInfo {
    int fd;
    size_t size;
    int rank;

    IpcInfo() : fd(-1), size(0), rank(-1) {}
    IpcInfo(int fd, size_t size, int rank) : fd(fd), size(size), rank(rank) {}
};

void send_fd(int sock, int fd, int rank, size_t size) {
    iovec iov[1];
    msghdr msg;
    auto rank_size = std::make_pair(rank, size);

    iov[0].iov_base = &rank_size;
    iov[0].iov_len = sizeof(rank_size);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_name = nullptr;
    msg.msg_namelen = 0;

    exchange_fd cmsg(SOL_SOCKET, SCM_RIGHTS, fd);
    msg.msg_control = &cmsg;
    msg.msg_controllen = sizeof(exchange_fd);

    EP_HOST_ASSERT(sendmsg(sock, &msg, 0) != -1 && "sendmsg failed");
}

IpcInfo recv_fd(int sock) {
    iovec iov[1];
    msghdr msg;
    std::pair<int, size_t> rank_size;

    iov[0].iov_base = &rank_size;
    iov[0].iov_len = sizeof(rank_size);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_name = nullptr;
    msg.msg_namelen = 0;

    exchange_fd cmsg;
    msg.msg_control = &cmsg;
    msg.msg_controllen = sizeof(exchange_fd);

    EP_HOST_ASSERT(recvmsg(sock, &msg, 0) != -1 && "recvmsg failed");
    return IpcInfo(cmsg.fd, rank_size.second, rank_size.first);
}

int create_server_socket(const char* sockname) {
    unlink(sockname);

    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sockname, sizeof(addr.sun_path) - 1);

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    EP_HOST_ASSERT(sock != -1 && "socket creation failed");

    auto size = offsetof(sockaddr_un, sun_path) + strlen(addr.sun_path);
    EP_HOST_ASSERT(bind(sock, (sockaddr*)&addr, size) != -1 && "bind failed");
    EP_HOST_ASSERT(listen(sock, 10) != -1 && "listen failed");

    return sock;
}

int connect_to_server(const char* sockname) {
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sockname, sizeof(addr.sun_path) - 1);

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    EP_HOST_ASSERT(sock != -1 && "socket creation failed");

    auto len = offsetof(sockaddr_un, sun_path) + strlen(addr.sun_path);
    for (int i = 0; i < 50; i++) {
        if (connect(sock, (sockaddr*)&addr, len) == 0)
            return sock;
        usleep(100000);  // 100ms
    }

    close(sock);
    EP_HOST_ASSERT(false && "connect failed after retries");
    return -1;
}

}  // anonymous namespace

std::vector<std::optional<pybind11::bytearray>> Buffer::all_gather_handle(
    const pybind11::bytearray& local_ipc_handle,
    const pybind11::function& barrier_func) {

    std::vector<std::optional<pybind11::bytearray>> result(num_ipc_ranks);

    const char* local_data = PyByteArray_AS_STRING(local_ipc_handle.ptr());
    size_t local_size = PyByteArray_GET_SIZE(local_ipc_handle.ptr());
    EP_HOST_ASSERT(local_size == sizeof(shared_memory::MemHandle) && "Invalid IPC handle size");

    const shared_memory::MemHandle* local_mem_handle =
        reinterpret_cast<const shared_memory::MemHandle*>(local_data);

    int my_fd = *reinterpret_cast<const int*>(&local_mem_handle->inner.ze_ipc_mem_handle);
    size_t my_size = local_mem_handle->size;

    // Store all IPC info
    std::vector<IpcInfo> all_ipc_handles(num_ipc_ranks);
    all_ipc_handles[ipc_rank] = IpcInfo(my_fd, my_size, ipc_rank);

    // Ring topology
    int dst_rank = (ipc_rank + 1) % num_ipc_ranks;

    // Get username for socket naming
    uid_t uid = getuid();
    struct passwd* pwd = getpwuid(uid);
    const char* username = pwd ? pwd->pw_name : "unknown";

    // Create server socket
    char server_name[256];
    snprintf(server_name, sizeof(server_name),
             "/tmp/deep_ep_ipc_ring_rank_%d_%s", ipc_rank, username);

    int server_sock = create_server_socket(server_name);
    barrier_func();

    // Connect to next rank
    char dst_server[256];
    snprintf(dst_server, sizeof(dst_server),
             "/tmp/deep_ep_ipc_ring_rank_%d_%s", dst_rank, username);
    int send_sock = connect_to_server(dst_server);
    barrier_func();

    // Accept from previous rank
    int recv_sock = accept(server_sock, nullptr, nullptr);
    EP_HOST_ASSERT(recv_sock != -1 && "accept failed");
    barrier_func();

    // Ring AllGather: N-1 steps
    IpcInfo current_ipc = all_ipc_handles[ipc_rank];
    for (int step = 1; step < num_ipc_ranks; ++step) {
        send_fd(send_sock, current_ipc.fd, current_ipc.rank, current_ipc.size);
        IpcInfo received = recv_fd(recv_sock);
        all_ipc_handles[received.rank] = received;
        current_ipc = received;
        barrier_func();
    }

    // Cleanup sockets
    close(send_sock);
    close(recv_sock);
    close(server_sock);
    unlink(server_name);

    // Convert to pybind11::bytearray
    for (int i = 0; i < num_ipc_ranks; ++i) {
        if (i == ipc_rank) {
            result[i] = pybind11::bytearray(local_data, local_size);
        } else {
            shared_memory::MemHandle mem_handle;
            memset(&mem_handle, 0, sizeof(mem_handle));
            *reinterpret_cast<int*>(&mem_handle.inner.ze_ipc_mem_handle) = all_ipc_handles[i].fd;
            mem_handle.size = all_ipc_handles[i].size;
            result[i] = pybind11::bytearray(reinterpret_cast<const char*>(&mem_handle), sizeof(mem_handle));
        }
    }

    return result;
}

template <int kNumRanks>
class CachedNotifyCombineKernel {
public:
    CachedNotifyCombineKernel(
        void** buffer_ptrs,
        int* send_head,
        int num_channels,
        int num_recv_tokens,
        int num_memset_int,
        int** barrier_signal_ptrs,
        int rank)
        : buffer_ptrs_(buffer_ptrs),
          send_head_(send_head),
          num_channels_(num_channels),
          num_recv_tokens_(num_recv_tokens),
          num_memset_int_(num_memset_int),
          barrier_signal_ptrs_(barrier_signal_ptrs),
          rank_(rank) {}

    SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const {
        auto sm_id = static_cast<int>(item.get_group(0));
        auto thread_id = static_cast<int>(item.get_local_id(0));
        auto num_threads = static_cast<int>(item.get_local_range(0));

        if (sm_id == 0) {
            // Block 0: Clean IPC buffer

            // Barrier before cleaning
            barrier_block_bypass<kNumRanks, true>(barrier_signal_ptrs_, rank_, item);

            // Clean buffer
            auto ptr = static_cast<int*>(buffer_ptrs_[rank_]);
            #pragma unroll
            for (int i = thread_id; i < num_memset_int_; i += num_threads)
                ptr[i] = 0;

            // Barrier after cleaning
            barrier_block_bypass<kNumRanks>(barrier_signal_ptrs_, rank_, item);
        } else {
            // Block 1 ~ num_channels: Fill in send_head array
            const auto channel_id = sm_id - 1;
            const auto rank_id = thread_id / 32;
            const auto lane_id = thread_id % 32;
            
            // Return if rank_id is out of range
            if (rank_id >= kNumRanks)
                return;

            // Get the token range this channel is responsible for
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens_, num_channels_, channel_id, token_start_idx, token_end_idx);

            // NOTES: `1 << 25` is a heuristic large number
            int last_head = 1 << 25;
            
            // Traverse from back to front, filling in reasonable values for negative send_head entries
            #pragma unroll
            for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
                int token_idx = token_idx_tail - lane_id;
                int expected_head = 0;
                
                // Read current head value; set to -1 if token_idx is out of bounds
                auto current_head = (token_idx >= token_start_idx) ? 
                    ld_nc_global(send_head_ + token_idx * kNumRanks + rank_id) : -1;
                
                // Process each token sequentially within the warp
                auto sg = item.get_sub_group();
                int num_iters = sycl::min(32, token_idx_tail - token_start_idx + 1);
                for (int i = 0; i < num_iters; ++i) {
                    // Broadcast head value from lane i
                    const int head = sycl::select_from_group(sg, current_head, i);
                    if (head < 0) {
                        // If head is negative, current lane needs to compute expected_head
                        if (lane_id == i)
                            expected_head = -last_head - 1;
                    } else {
                        // Update last_head
                        last_head = head;
                    }
                }
                
                // If current_head is negative and token_idx is valid, write back expected_head
                if (current_head < 0 && token_idx >= token_start_idx)
                    send_head_[token_idx * kNumRanks + rank_id] = expected_head;
            }
        }
    }

private:
    void** buffer_ptrs_;
    int* send_head_;
    int num_channels_;
    int num_recv_tokens_;
    int num_memset_int_;
    int** barrier_signal_ptrs_;
    int rank_;
};

namespace intranode {

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           sycl::queue& stream) {
    
    // Thread count: at least 128, each rank needs 32 threads (one warp)
    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);

    // grid size: 1 + num_channels
    // - Block 0: Clean IPC buffer
    // - Block 1 ~ num_channels: Fill in send_head array
    int num_blocks = 1 + num_channels;
    sycl::range<1> global_range(num_blocks * num_threads);
    sycl::range<1> local_range(num_threads);

    #define CACHED_NOTIFY_COMBINE_LAUNCH_CASE(ranks)                                    \
        case ranks:                                                                     \
            stream.submit([&](sycl::handler& cgh) {                                \
                                                                                                        cgh.parallel_for(                                                      \
                    sycl::nd_range<1>(global_range, local_range),                     \
                    [=](sycl::nd_item<1> item) {                                      \
                        CachedNotifyCombineKernel<ranks> kernel(                       \
                            buffer_ptrs,                                               \
                            send_head,                                                 \
                            num_channels,                                              \
                            num_recv_tokens,                                           \
                            num_memset_int,                                            \
                            barrier_signal_ptrs,                                       \
                            rank);                                                     \
                        kernel(item);                                                  \
                    });                                                                \
            });                                                                        \
            break

    switch (num_ranks) {
        CACHED_NOTIFY_COMBINE_LAUNCH_CASE(2);
        CACHED_NOTIFY_COMBINE_LAUNCH_CASE(4);
        CACHED_NOTIFY_COMBINE_LAUNCH_CASE(8);
        default:
            EP_HOST_ASSERT(false && "Unsupported number of ranks");
    }

    #undef CACHED_NOTIFY_COMBINE_LAUNCH_CASE
    
    stream.wait();
}

}  // namespace intranode

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<deep_ep::EventHandle>>
deep_ep::Buffer::intranode_combine(const torch::Tensor& x,
                                  const std::optional<torch::Tensor>& topk_weights,
                                  const std::optional<torch::Tensor>& bias_0,
                                  const std::optional<torch::Tensor>& bias_1,
                                  const torch::Tensor& src_idx,
                                  const torch::Tensor& rank_prefix_matrix,
                                  const torch::Tensor& channel_prefix_matrix,
                                  const torch::Tensor& send_head,
                                  const deep_ep::Config& config,
                                  std::optional<deep_ep::EventHandle>& previous_event,
                                  bool async,
                                  bool allocate_on_comm_stream) {
    // Validate input tensors
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel uses two blocks: even-numbered for sending, odd-numbered for receiving.
    EP_HOST_ASSERT(config.num_eus % 2 == 0);
    int num_channels = config.num_eus / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));  // Original token count
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Handle topk weights
    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ipc_ranks * sizeof(int) * 2 <= num_ipc_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ipc_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     ipc_rank,
                                     num_ipc_ranks,
                                     comm_stream);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Allocate output tensor with correct size (num_recv_tokens is the original token count)
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ipc_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ipc_ranks * config.num_max_ipc_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * num_ipc_ranks * config.num_max_ipc_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * num_ipc_ranks * config.num_max_ipc_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_ipc_bytes);

    // Map torch dtype to SYCL DataType enum
    DataType sycl_dtype;
    switch (x.scalar_type()) {
        case torch::kBFloat16: sycl_dtype = DataType::kBFloat16; break;
        case torch::kInt32:    sycl_dtype = DataType::kInt32;    break;
        default: EP_HOST_ASSERT(false && "Unsupported dtype for combine");
    }

    // Call combine kernel
    intranode::combine(sycl_dtype,
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       ipc_rank,
                       num_ipc_ranks,
                       comm_stream,
                       config.num_eus,
                       config.num_max_ipc_chunked_send_tokens,
                       config.num_max_ipc_chunked_recv_tokens);

    // Wait for completion if not async
    if (!async) {
        comm_stream.wait();
    }

    return std::make_tuple(recv_x, recv_topk_weights, std::nullopt);
}


}  // namespace deep_ep


namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP XPU: expert-parallel communication library for Intel XPU";

    py::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    py::class_<deep_ep::Config>(m, "Config")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_eus") = 20,
             py::arg("num_max_ipc_chunked_send_tokens") = 6,
             py::arg("num_max_ipc_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 0,
             py::arg("num_max_rdma_chunked_recv_tokens") = 0)
        .def_readonly("num_eus", &deep_ep::Config::num_eus)
        .def_readonly("num_max_ipc_chunked_send_tokens", &deep_ep::Config::num_max_ipc_chunked_send_tokens)
        .def_readonly("num_max_ipc_chunked_recv_tokens", &deep_ep::Config::num_max_ipc_chunked_recv_tokens)
        .def_readonly("num_max_rdma_chunked_send_tokens", &deep_ep::Config::num_max_rdma_chunked_send_tokens)
        .def_readonly("num_max_rdma_chunked_recv_tokens", &deep_ep::Config::num_max_rdma_chunked_recv_tokens)
        .def("get_ipc_buffer_size_hint", &deep_ep::Config::get_ipc_buffer_size_hint);


    py::class_<deep_ep::Buffer>(m, "Buffer")
        .def(py::init<int, int, int64_t, int64_t, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_comm_queue", &deep_ep::Buffer::get_comm_queue)
        .def("all_gather_handle", &deep_ep::Buffer::all_gather_handle,
             py::arg("local_ipc_handle"),
             py::arg("barrier_func"))
        .def("test_barrier", &deep_ep::Buffer::test_barrier,
             py::arg("process_group") = std::nullopt,
             "Test barrier synchronization across GPUs. Optionally provide a ProcessGroup for CPU barriers.")
        .def("test_barrier_stress", &deep_ep::Buffer::test_barrier_stress,
             py::arg("inner_repeat"),
             py::arg("iter_offset"),
             py::arg("data_size"),
             "Barrier stress test with IPC data verification. Returns error count.")
        .def("test_barrier_perf", &deep_ep::Buffer::test_barrier_perf,
             py::arg("inner_repeat"),
             "Pure barrier performance test (no data verification).")
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout,
             py::arg("topk_idx"),
             py::arg("num_experts"),
             py::arg("previous_event"),
             py::arg("async_op") = false,
             py::arg("allocate_on_comm_stream") = false,
             "Compute dispatch layout: token distribution across experts and ranks.")
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch,
             py::arg("x"),
             py::arg("x_scales"),
             py::arg("topk_idx"),
             py::arg("topk_weights"),
             py::arg("num_tokens_per_rank"),
             py::arg("is_token_in_rank"),
             py::arg("num_tokens_per_expert"),
             py::arg("cached_num_recv_tokens"),
             py::arg("cached_rank_prefix_matrix"),
             py::arg("cached_channel_prefix_matrix"),
             py::arg("expert_alignment"),
             py::arg("num_worst_tokens"),
             py::arg("config"),
             py::arg("previous_event"),
             py::arg("async_op") = false,
             py::arg("allocate_on_comm_stream") = false)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine,
             py::arg("x"),
             py::arg("topk_weights"),
             py::arg("bias_0"),
             py::arg("bias_1"),
             py::arg("src_idx"),
             py::arg("rank_prefix_matrix"),
             py::arg("channel_prefix_matrix"),
             py::arg("send_head"),
             py::arg("config"),
             py::arg("previous_event"),
             py::arg("async_op") = false,
             py::arg("allocate_on_comm_stream") = false)
        .def("test_notify_dispatch", &deep_ep::Buffer::test_notify_dispatch,
             py::arg("num_tokens_per_rank"),
             py::arg("num_tokens_per_expert"),
             py::arg("is_token_in_rank"),
             py::arg("num_tokens"),
             py::arg("num_experts"),
             py::arg("num_channels"),
             py::arg("expert_alignment"),
             "Test notify_dispatch kernel: exchange token counts and compute prefix matrices.");
}


void deep_ep::Buffer::test_barrier_perf(int inner_repeat) {
    EP_HOST_ASSERT(is_available() && "Buffer must be synced before calling test_barrier_perf");
    EP_HOST_ASSERT(inner_repeat > 0 && "inner_repeat must be positive");

    intranode::barrier_perf_test(
        barrier_signal_ptrs_gpu,
        ipc_rank, num_ipc_ranks,
        inner_repeat,
        comm_stream);
}

int deep_ep::Buffer::test_barrier_stress(int inner_repeat, int iter_offset, int data_size) {
    EP_HOST_ASSERT(is_available() && "Buffer must be synced before calling test_barrier_stress");
    EP_HOST_ASSERT(data_size > 0 && "data_size must be positive");

    // Use the tail of the IPC data buffer for stress test data
    // data_offset_ints is the int-offset from the start of each rank's buffer
    int data_bytes = data_size * sizeof(int);
    EP_HOST_ASSERT(data_bytes <= num_ipc_bytes && "data_size too large for IPC buffer");
    int data_offset_ints = static_cast<int>((num_ipc_bytes - data_bytes) / sizeof(int));

    // Allocate error_count on device (single int, zeroed)
    int* error_count_dev = nullptr;
    error_count_dev = sycl::malloc_device<int>(1, comm_stream);
    comm_stream.memset(error_count_dev, 0, sizeof(int)).wait();

    // Launch kernel
    intranode::barrier_stress_test(
        buffer_ptrs_gpu, barrier_signal_ptrs_gpu,
        error_count_dev,
        ipc_rank, num_ipc_ranks,
        inner_repeat, iter_offset,
        data_size, data_offset_ints,
        comm_stream);

    // Copy error count back
    int error_count = 0;
    comm_stream.memcpy(&error_count, error_count_dev, sizeof(int)).wait();
    sycl::free(error_count_dev, comm_stream);

    return error_count;
}

void deep_ep::Buffer::test_barrier(const std::optional<c10::intrusive_ptr<c10d::ProcessGroup>>& process_group) {
    EP_HOST_ASSERT(is_available() && "Buffer must be synced before calling test_barrier");
    
    // CPU barrier before GPU barrier
    if (process_group.has_value()) {
        auto work = process_group.value()->barrier();
        work->wait();
    }
    
    // GPU barrier
    intranode::barrier(barrier_signal_ptrs_gpu, ipc_rank, num_ipc_ranks, comm_stream);
    
    // CPU barrier after GPU barrier
    if (process_group.has_value()) {
        auto work = process_group.value()->barrier();
        work->wait();
    }
}


std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<deep_ep::EventHandle>>
deep_ep::Buffer::get_dispatch_layout(const torch::Tensor& topk_idx,
                                     int num_experts,
                                     std::optional<deep_ep::EventHandle>& previous_event,
                                     bool async,
                                     bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, torch::dtype(torch::kInt32).device(torch::kXPU));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, torch::dtype(torch::kInt32).device(torch::kXPU));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, torch::dtype(torch::kBool).device(torch::kXPU));

    deep_ep::layout::get_dispatch_layout(topk_idx.data_ptr<deep_ep::topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);
    return std::make_tuple(num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, std::nullopt);
}


std::tuple<int, std::vector<int>, torch::Tensor, torch::Tensor> 
deep_ep::Buffer::test_notify_dispatch(
    const torch::Tensor& num_tokens_per_rank,
    const torch::Tensor& num_tokens_per_expert,
    const torch::Tensor& is_token_in_rank,
    int num_tokens,
    int num_experts,
    int num_channels,
    int expert_alignment) {
    
    EP_HOST_ASSERT(is_available() && "Buffer must be synced before calling test_notify_dispatch");
    
    // Validate inputs
    EP_HOST_ASSERT(num_tokens_per_rank.dim() == 1 && num_tokens_per_rank.size(0) == num_ipc_ranks);
    EP_HOST_ASSERT(num_tokens_per_expert.dim() == 1 && num_tokens_per_expert.size(0) == num_experts);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 && is_token_in_rank.size(0) == num_tokens && is_token_in_rank.size(1) == num_ipc_ranks);
    
    int num_local_experts = num_experts / num_ipc_ranks;
    
    // Allocate output tensors
    auto rank_prefix_matrix = torch::empty({num_ipc_ranks, num_ipc_ranks}, torch::dtype(torch::kInt32).device(torch::kXPU));
    auto channel_prefix_matrix = torch::empty({num_ipc_ranks, num_channels}, torch::dtype(torch::kInt32).device(torch::kXPU));
    
    // Reset counters
    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter_mapped[i] = -1;
    
    // Calculate memset size
    int num_memset_int = num_channels * num_ipc_ranks * 4;
    
    // Call notify_dispatch kernel (synchronous — stream.wait() inside)
    intranode::notify_dispatch(
        num_tokens_per_rank.data_ptr<int>(),
        moe_recv_counter_mapped,
        num_ipc_ranks,
        num_tokens_per_expert.data_ptr<int>(),
        moe_recv_expert_counter_mapped,
        num_experts,
        num_tokens,
        is_token_in_rank.data_ptr<bool>(),
        channel_prefix_matrix.data_ptr<int>(),
        rank_prefix_matrix.data_ptr<int>(),
        num_memset_int,
        expert_alignment,
        buffer_ptrs_gpu,
        barrier_signal_ptrs_gpu,
        ipc_rank,
        comm_stream,
        num_channels);
    
    // Read results (kernel is already synchronous, no polling needed)
    int num_recv_tokens = static_cast<int>(*moe_recv_counter);
    std::vector<int> expert_counts(num_local_experts);
    for (int i = 0; i < num_local_experts; ++i)
        expert_counts[i] = moe_recv_expert_counter_mapped[i];
    
    return std::make_tuple(num_recv_tokens, expert_counts, rank_prefix_matrix, channel_prefix_matrix);
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<deep_ep::EventHandle>>
deep_ep::Buffer::intranode_dispatch(const torch::Tensor& x,
                                   const std::optional<torch::Tensor>& x_scales,
                                   const std::optional<torch::Tensor>& topk_idx,
                                   const std::optional<torch::Tensor>& topk_weights,
                                   const std::optional<torch::Tensor>& num_tokens_per_rank,
                                   const torch::Tensor& is_token_in_rank,
                                   const std::optional<torch::Tensor>& num_tokens_per_expert,
                                   int cached_num_recv_tokens,
                                   const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                                   const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                                   int expert_alignment,
                                   int num_worst_tokens,
                                   const deep_ep::Config& config,
                                   std::optional<deep_ep::EventHandle>& previous_event,
                                   bool async,
                                   bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel uses two work-groups: even for sending, odd for receiving
    EP_HOST_ASSERT(config.num_eus % 2 == 0);
    int num_channels = config.num_eus / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 && is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) && is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 && cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks && cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 && cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks && cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 && num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 && num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 && topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) && num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 || x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, ipc_rank, num_ipc_ranks, comm_stream);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, torch::dtype(torch::kInt32).device(torch::kXPU));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, torch::dtype(torch::kInt32).device(torch::kXPU));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter_mapped[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_ipc_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ipc_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   ipc_rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts && ready; ++i)
                    ready &= moe_recv_expert_counter_mapped[i] >= 0;

                if (ready)
                    break;
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter_mapped, moe_recv_expert_counter_mapped + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, torch::dtype(torch::kInt32).device(torch::kXPU));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, torch::dtype(torch::kInt32).device(torch::kXPU));
    auto send_head = torch::empty({num_tokens, num_ranks}, torch::dtype(torch::kInt32).device(torch::kXPU));

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
            num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
            num_channels * num_ranks * config.num_max_ipc_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_ipc_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
            num_channels * num_ranks * config.num_max_ipc_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
            num_channels * num_ranks * config.num_max_ipc_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
            num_channels * num_ranks * config.num_max_ipc_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
        <= num_ipc_bytes);
    intranode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        ipc_rank,
                        num_ipc_ranks,
                        comm_stream,
                        config.num_eus,
                        config.num_max_ipc_chunked_send_tokens,
                        config.num_max_ipc_chunked_recv_tokens);

    // Wait for completion if not async
    if (!async) {
        comm_stream.wait();
    }

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            std::nullopt};
}
