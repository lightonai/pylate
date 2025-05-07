from dataclasses import dataclass

from .base_config import BaseConfig
from .settings import (
    DocSettings,
    IndexingSettings,
    QuerySettings,
    ResourceSettings,
    RunSettings,
    SearchSettings,
    TokenizerSettings,
    TrainingSettings,
)


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(
    RunSettings,
    ResourceSettings,
    DocSettings,
    QuerySettings,
    TrainingSettings,
    IndexingSettings,
    SearchSettings,
    BaseConfig,
    TokenizerSettings,
):
    pass
