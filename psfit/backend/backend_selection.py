import os

import torch.cuda

from psfit.backend import numpy_backend, pytorch_backend


def get_backend():
    backend_mapping = {
        "numpy": numpy_backend,
        "pytorch": pytorch_backend,
    }

    bend = os.environ.get("backend")

    if bend is None:
        # print("backend not specified. numpy is selected as the default backend.")
        return numpy_backend()

    backend_function = backend_mapping.get(bend)

    if backend_function is None:
        raise ValueError(
            f"Unknown default_backend '{bend}'. Available options are: {', '.join(backend_mapping.keys())}.")

    if bend == "pytorch":
        if torch.cuda.is_available():
            # print(f"{bend} with cuda support is selected as the backend.")
            return backend_function()

        raise ValueError(f"cuda is not available.")

    # print(f"{bend} is selected as the backend.")
    return backend_function()
