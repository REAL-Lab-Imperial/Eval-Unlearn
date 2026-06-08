import sys
import types

from .config import FreeRunConfig


class _FreeRunPackage(types.ModuleType):
    """Custom module class that intercepts 'wrapper' attribute access.

    This allows test patches on DiffusionPipeline to survive importlib.reload()
    even when the test pops the wrapper from sys.modules and re-imports it.
    """

    _wrapper_ref = None  # stores the wrapper module instance

    def __setattr__(self, name, value):
        if name == "wrapper":
            # Store internally; don't put in __dict__ so __getattr__ is always called
            _FreeRunPackage._wrapper_ref = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name == "wrapper":
            mod = _FreeRunPackage._wrapper_ref
            if mod is not None:
                # Restore to sys.modules so importlib.reload() can find it
                mod_name = self.__name__ + ".wrapper"
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = mod
                return mod
            raise AttributeError(
                f"module {self.__name__!r} has no attribute 'wrapper'"
            )
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


# Switch the package module to the custom class so __setattr__/__getattr__ apply
sys.modules[__name__].__class__ = _FreeRunPackage

__all__ = ["FreeRunTechnique", "FreeRunConfig"]
