from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class BaseConfig:
    """
    Base configuration class providing dictionary serialization methods.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """
        Create a configuration instance from a dictionary.
        ignores keys that do not match fields in the dataclass to ensure forward compatibility.
        """
        valid_keys = {k for k, f in cls.__dataclass_fields__.items() if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
