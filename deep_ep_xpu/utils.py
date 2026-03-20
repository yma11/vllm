import torch
from typing import Any, Optional

# noinspection PyUnresolvedReferences
from deep_ep_cpp import EventHandle


class EventOverlap:
    """
    A wrapper class to manage XPU events, also for better overlapping convenience.

    Attributes:
        event: the XPU event captured.
    """

    def __init__(self, event: Optional[EventHandle] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the XPU event captured.
        """
        self.event = event

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.xpu.current_stream()` waits for the event to be finished.
        """
        if self.event is None:
            raise RuntimeError("EventOverlap.current_stream_wait() called but no event is set.")
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap:
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream will wait for the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()
