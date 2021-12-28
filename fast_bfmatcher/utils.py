import time
from typing import Any, Dict, Optional


class measuretime:
    def __init__(
        self,
        name: str,
        extra: Optional[Dict[str, Any]] = None,
        log: bool = True,
        num_steps: int = 1,
    ):
        self.name = name
        self.extra = extra
        self.log = log
        self.num_steps = num_steps

    @property
    def params(self) -> str:
        if self.extra is None:
            return ""

        params = []
        for k in sorted(self.extra):
            params.append(f"{k}={self.extra[k]}")
        params = ", ".join(params)

        return f"PARAMS: {params}"

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.seconds = time.perf_counter() - self.t
        if self.log:
            print(
                f"{self.name}: {self.num_steps} steps took {self.seconds:6.3f} [s], "
                f"per step {1000 * self.seconds / self.num_steps:6.3f} [ms]"
                f" {self.params}"
            )
