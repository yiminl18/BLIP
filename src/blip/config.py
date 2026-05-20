from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DeploymentConfig:
    api_key: str
    api_version: str
    azure_endpoint: str
    deployment: str
    model_name: str = ""


@dataclass(frozen=True)
class AzureConfig:
    driver: DeploymentConfig
    escalation: DeploymentConfig
    embed: DeploymentConfig


def _load_key_file(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        result[k.strip()] = v.strip()
    return result


def _deployment_from_file(path: str | Path) -> DeploymentConfig:
    d = _load_key_file(path)
    return DeploymentConfig(
        api_key=d["api_key"],
        api_version=d.get("api_version", "2024-12-01-preview"),
        azure_endpoint=d["azure_endpoint"],
        deployment=d.get("deployment", ""),
        model_name=d.get("model_name", ""),
    )


@dataclass(frozen=True)
class Config:
    azure: AzureConfig
    cache_dir: Path
    seed: int = 42
    block_count_m: int = 20
    refine_threshold_t: int = 10
    f_L_driver: int = 2
    f_L_escalation: int = 2
    judge_prompt: str = "llm_equal_human_example"
    judge_model: str = "driver"
    fastpath: str = "refine"


_REPO_ROOT = Path(__file__).parent.parent.parent


def load_config(azure_json: Path | None = None) -> Config:
    if azure_json is None:
        azure_json = _REPO_ROOT / "local" / "azure.json"
    spec = json.loads(azure_json.read_text())
    driver = _deployment_from_file(spec["key_file_cheap"])
    escalation = _deployment_from_file(spec["key_file"])
    embed = _deployment_from_file(spec["embedding_key_file"])
    azure = AzureConfig(driver=driver, escalation=escalation, embed=embed)
    return Config(azure=azure, cache_dir=_REPO_ROOT / "cache")


# module-level singleton loaded lazily
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
