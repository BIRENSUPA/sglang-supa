"""Shared utilities for SUPA kernels.ops adaptations."""

from __future__ import annotations


class SupaApiWrapper:
    """Adapt a SUPA API while retaining the wrapped kernel for debugging."""

    def __init__(self, kernel, adapt_grid=None, adapt_args=None):
        self._kernel = kernel
        self._kernel_name = getattr(
            kernel,
            "__qualname__",
            getattr(kernel, "__name__", type(kernel).__name__),
        )
        self.__wrapped__ = kernel
        self._adapt_grid = adapt_grid
        self._adapt_args = adapt_args

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self._adapt_args:
                self._adapt_args(args, kwargs)
            launch_grid = (
                self._adapt_grid(grid, args, kwargs) if self._adapt_grid else grid
            )
            return self._kernel[launch_grid](*args, **kwargs)

        return launch

    def __call__(self, *args, **kwargs):
        if self._adapt_args:
            self._adapt_args(args, kwargs)
        return self._kernel(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __repr__(self):
        return (
            f"{type(self).__name__}(kernel={self._kernel_name}, "
            f"adapt_grid={self._adapt_grid!r}, adapt_args={self._adapt_args!r})"
        )
