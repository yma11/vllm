#pragma once

#include <memory>
#include <torch/types.h>
#include <ATen/xpu/XPUContext.h>
#include "sycl/configs.h"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kXPU);
        event->record(at::xpu::getCurrentXPUStream());
    }

    explicit EventHandle(const at::xpu::XPUStream& stream) {
        event = std::make_shared<torch::Event>(torch::kXPU);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { event->block(at::xpu::getCurrentXPUStream()); }
};

inline torch::Event create_event(const at::xpu::XPUStream& s) {
    auto event = torch::Event(torch::kXPU);
    event.record(s);
    return event;
}

inline void stream_wait(const at::xpu::XPUStream& s_0, const at::xpu::XPUStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    auto event = create_event(s_1);
    event.block(s_0);
}

inline void stream_wait(const at::xpu::XPUStream& s, const EventHandle& event) {
    event.event->block(s);
}

}  // namespace deep_ep
