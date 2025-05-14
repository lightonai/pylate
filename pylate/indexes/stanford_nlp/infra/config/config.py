from dataclasses import dataclass

from .base_config import BaseConfig
from .settings import (
    DocSettings,
    IndexingSettings,
    ResourceSettings,
    RunSettings,
    SearchSettings,
)


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(
    RunSettings,
    ResourceSettings,
    DocSettings,
    IndexingSettings,
    SearchSettings,
    BaseConfig,
):
    pass
