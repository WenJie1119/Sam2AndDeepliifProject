"""
Device helpers shared by CLI entrypoints.
"""

import torch


def prepare_device(device_arg: str) -> str:
    """
    Normalize CLI device input and set the current CUDA device when needed.

    Accepted inputs:
    - "cpu"
    - "cuda"
    - "cuda:N"
    - "N"  (normalized to "cuda:N")
    """
    device = str(device_arg).strip().lower()
    if not device:
        raise ValueError("--device cannot be empty")

    if device == "cpu":
        return "cpu"

    if device.isdigit():
        device = f"cuda:{int(device)}"
    elif device == "cuda":
        device = "cuda:0"

    if not device.startswith("cuda:"):
        raise ValueError(
            f"Unsupported --device '{device_arg}'. Use 'cpu', 'cuda', 'cuda:N', or 'N'."
        )

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        return "cpu"

    parsed = torch.device(device)
    device_index = 0 if parsed.index is None else parsed.index
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise ValueError(
            f"Requested CUDA device {device_index}, but only {device_count} visible GPU(s) are available."
        )

    # Third-party code inside SAM2 still uses bare 'cuda' in a few places.
    # Set the current device so those allocations land on the requested GPU.
    torch.cuda.set_device(device_index)
    return f"cuda:{device_index}"
