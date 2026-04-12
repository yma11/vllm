#include "deep_ep.hpp"

#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <chrono>
#include <iostream>
#include <memory>

#include "sycl/api.hpp"
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include "sycl/configs.h"

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

    // MoE counters (shared memory for host-device access)
    moe_recv_counter = const_cast<volatile int*>(
        static_cast<int*>(sycl::malloc_shared(sizeof(int64_t), comm_stream)));
    moe_recv_counter_mapped = const_cast<int*>(moe_recv_counter);
    *moe_recv_counter = -1;

    moe_recv_expert_counter = const_cast<volatile int*>(
        static_cast<int*>(sycl::malloc_shared(sizeof(int) * NUM_MAX_LOCAL_EXPERTS, comm_stream)));
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
             py::arg("barrier_func"));
}
