from importlib.util import find_spec

from vllm.logger import init_logger

from vllm.platforms import current_platform

logger = init_logger(__name__)

HAS_TRITON = find_spec("triton") is not None and not current_platform.is_xpu()

if not HAS_TRITON:
    logger.info("Triton not installed or compatible; certain GPU-related functions"
                " will not be available.")
