"""Lazy BF16 + LoRA adapter matching the selected Sol round-2 offline model.

Reuses the owned-child JSONL transport, not InternVL's model/preprocessing.
Construction is CPU-only. An explicit physical GPU is mandatory.
"""

from pathlib import Path

from mobile_world.reflection.gui_reflection import GUIReflectionReflector, ROOT
from mobile_world.reflection.protocol import protocol_configuration

DEFAULT_ADAPTER = ROOT / "artifacts/training/qreflector_sol_round2_geometry_20261009_gpu2/train/final"
DEFAULT_BASE = ROOT / "artifacts/models/qreflector_base_bf16/Qwen3-VL-8B-Instruct"
DEFAULT_PROCESSOR = ROOT / "artifacts/training/qreflector_sol_round1_v9_epoch9_20261008_gpu7/train/final"
DEFAULT_GENERATION_CONFIG = ROOT / "artifacts/models/qreflector_base/qwen3-vl-8b-instruct-unsloth-bnb-4bit"
DEFAULT_PYTHON = ROOT / "artifacts/envs/qreflector-native/bin/python"


class QReflectorV9(GUIReflectionReflector):
    name = "qreflector_v9"

    def __init__(self, *, gpu_uuid, adapter_path=None, python_path=None,
                 base_path=None, processor_path=None, generation_config_path=None,
                 companion_process_manifest=None):
        if not gpu_uuid:
            raise ValueError("Q-Reflector v9 requires an explicit physical GPU UUID")
        super().__init__(model_path=adapter_path or DEFAULT_ADAPTER,
                         python_path=python_path or DEFAULT_PYTHON,
                         gpu_uuid=gpu_uuid, protocol="v9_geometry", request_timeout=300)
        self.base_path = str(Path(base_path or DEFAULT_BASE).absolute())
        self.processor_path = str(Path(processor_path or DEFAULT_PROCESSOR).absolute())
        self.generation_config_path = str(Path(generation_config_path or DEFAULT_GENERATION_CONFIG).absolute())
        self.companion_process_manifest = (str(Path(companion_process_manifest).absolute())
                                           if companion_process_manifest else None)

    @property
    def configuration(self):
        return {**protocol_configuration(self.protocol), "model": self.name,
                "adapter_path": self.model_path, "base_path": self.base_path,
                "processor_path": self.processor_path, "generation_config_path": self.generation_config_path,
                "companion_process_manifest": self.companion_process_manifest,
                "python_path": self.python_path, "gpu_uuid": self.gpu_uuid,
                "dtype": "bfloat16", "quantization": None, "do_sample": False,
                "seed": 42, "context_limit": 8192, "memory_cap_gib": self.memory_cap_gib}

    def _worker_environment(self):
        env = super()._worker_environment()
        env["HF_HOME"] = str(ROOT / "artifacts/cache/qreflector-online-hf")
        return env

    def _worker_command(self):
        for path in (Path(self.python_path), Path(self.model_path) / "adapter_config.json",
                     Path(self.model_path) / "adapter_model.safetensors", Path(self.base_path) / "config.json",
                     Path(self.processor_path) / "preprocessor_config.json",
                     Path(self.processor_path) / "chat_template.jinja",
                     Path(self.generation_config_path) / "generation_config.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Q-Reflector v9 dependency missing: {path}")
        command = [self.python_path, "-u", str(Path(__file__).with_name("qreflector_v9_worker.py")),
                "--adapter", self.model_path, "--base", self.base_path,
                "--processor", self.processor_path, "--generation-config", self.generation_config_path,
                "--memory-cap-gib", str(self.memory_cap_gib),
                "--max-new-tokens", str(self.max_new_tokens)]
        if self.companion_process_manifest:
            command += ["--companion-process-manifest", self.companion_process_manifest]
        return command
