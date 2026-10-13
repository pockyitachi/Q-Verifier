#!/usr/bin/env python3
"""Prepare, audit, or explicitly train Q-Reflector with privileged-context OPSD.

No GPU is selected and no model is loaded at import time. See
docs/qreflector_opsd.md for the objective, limitations, and launch procedure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '../datasets/condition_generation_v6_user_reviewed'
BASE = ROOT / "artifacts/models/qreflector_base_bf16/Qwen3-VL-8B-Instruct"
FIELDS = ("precondition", "postcondition", "reason", "answer")
STUDENT = "default"
TEACHER = "opsd_teacher"
REFERENCE_MARKER = "=== TRAINING-ONLY REFERENCE FOR THE TEACHER ==="
SOURCE = "https://arxiv.org/html/2601.18734v3"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def target_of(row):
    if [m["role"] for m in row["messages"]] != ["user", "assistant"]:
        raise ValueError("Expected exactly one user input and one reference response")
    content = row["messages"][1]["content"]
    if len(content) != 1 or content[0]["type"] != "text":
        raise ValueError("Reference must be one JSON text block")
    target = json.loads(content[0]["text"])
    if set(target) != set(FIELDS) or target["answer"] not in ("Yes", "No"):
        raise ValueError("Invalid four-field reference")
    if not all(isinstance(v, str) and v.strip() for v in target.values()):
        raise ValueError("Empty or non-string reference field")
    return target


def build_messages(row, reference_fields="full"):
    """Never place reference fields in the student's input; do not mutate rows."""
    target = target_of(row)
    student = [copy.deepcopy(row["messages"][0])]
    if any(REFERENCE_MARKER in c.get("text", "") for c in student[0]["content"]):
        raise ValueError("Training reference marker already exists in student input")
    if reference_fields not in ("full", "answer"):
        raise ValueError("reference_fields must be full or answer")
    reference = target if reference_fields == "full" else {"answer": target["answer"]}
    teacher = copy.deepcopy(student)
    teacher[0]["content"].append({
        "type": "text",
        "text": "\n\n" + REFERENCE_MARKER + "\n"
        "The following is the reference annotation for this training example. "
        "Use it together with the task, action, history, and both screenshots to "
        "inform your verification. It is reference information, not an additional "
        "GUI observation.\n" + json.dumps(reference, ensure_ascii=False) + "\n"
        "=== END REFERENCE ===\n"
        "Now produce your own verification response in the requested four-field "
        "JSON format. Do not repeat these instructions or add a reference field.",
    })
    return student, teacher


def validate_training_data(data):
    """Read only train rows and their images; never open val/test rows or labels."""
    data = Path(data)
    frozen = read_json(data / "FROZEN.json")
    if frozen.get("status") != "frozen":
        raise ValueError("A frozen dataset is required")
    for name in ("train.jsonl", "prompt_template.txt", frozen["image_manifest"]):
        if digest(data / name) != frozen["artifact_sha256"][name]:
            raise ValueError(f"Frozen artifact changed: {name}")
    rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()
            if line.strip()]
    if len(rows) != 882 or frozen["splits"]["train"]["rows"] != 882:
        raise ValueError("This entry point is scoped to the frozen 882-row training split")
    image_manifest = read_json(data / frozen["image_manifest"])
    checked = {}
    labels = {"Yes": 0, "No": 0}
    for row in rows:
        labels[target_of(row)["answer"]] += 1
        student, _ = build_messages(row)
        paths = [c["image"] for c in student[0]["content"] if c["type"] == "image"]
        if len(paths) != 2:
            raise ValueError("Exactly two ordered screenshots are required")
        for name in paths:
            if name not in image_manifest:
                raise ValueError(f"Unregistered image: {name}")
            if name not in checked:
                checked[name] = digest(data / name)
            if checked[name] != image_manifest[name]["sha256"]:
                raise ValueError(f"Frozen image changed: {name}")
    if labels != frozen["splits"]["train"]["labels"]:
        raise ValueError("Training label counts changed")
    return rows, {"rows": len(rows), "labels": labels, "verified_image_paths": len(checked),
                  "train_sha256": digest(data / "train.jsonl"),
                  "frozen_sha256": digest(data / "FROZEN.json"),
                  "student_prompt": "Unmodified frozen v6 user message",
                  "validation_or_test_rows_read": False}


def validate_hyperparameters(config):
    positive = ("epochs", "gradient_accumulation", "learning_rate", "max_new_tokens",
                "max_sequence_length", "image_max_side", "save_every", "loss_chunk_tokens")
    for name in positive:
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["max_steps"] < 0 or config["sft_weight"] < 0:
        raise ValueError("max_steps and sft_weight must be nonnegative")
    if config["pointwise_clip"] < 0 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Invalid clipping or warmup configuration")
    if config["max_new_tokens"] >= config["max_sequence_length"]:
        raise ValueError("Generation limit leaves no room for the prompt")


def pinned_files(config):
    adapter, base = Path(config["adapter"]), Path(config["base"])
    names = [adapter / "adapter_model.safetensors", adapter / "adapter_config.json",
             adapter / "tokenizer.json", adapter / "tokenizer_config.json",
             adapter / "processor_config.json", base / "config.json"]
    for optional in ("chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (adapter / optional).exists():
            names.append(adapter / optional)
    if (base / "model.safetensors.index.json").exists():
        names.append(base / "model.safetensors.index.json")
    return {str(path): digest(path) for path in names}


def prepare(args):
    config = {key: value for key, value in vars(args).items() if key not in ("mode", "output")}
    for key in ("data", "base", "adapter"):
        config[key] = str(config[key].resolve())
    validate_hyperparameters(config)
    adapter_config = read_json(Path(config["adapter"]) / "adapter_config.json")
    if (adapter_config.get("peft_type") != "LORA" or adapter_config.get("bias", "none") != "none"
            or adapter_config.get("modules_to_save")):
        raise ValueError("Only bias-free LoRA adapters without modules_to_save are supported")
    if read_json(Path(config["base"]) / "config.json").get("model_type") != "qwen3_vl":
        raise ValueError("This implementation supports the Qwen3-VL BF16 base")
    rows, validation = validate_training_data(Path(config["data"]))
    hashes = pinned_files(config)
    steps = config["epochs"] * math.ceil(len(rows) / config["gradient_accumulation"])
    if config["max_steps"]:
        steps = min(steps, config["max_steps"])
    # Never repurpose an existing SFT/test run directory.
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"format_version": 1, "config": config, "data_validation": validation,
            "pinned_files": hashes, "worker_sha256": digest(__file__),
            "optimizer_steps": steps, "algorithm_source": SOURCE,
            "teacher_policy": "Frozen copy of the initial SFT LoRA, sharing frozen BF16 base",
            "objective": "Mean full-vocabulary forward KL with pointwise upper clipping, plus weighted SFT",
            "teacher_reference": config["reference_fields"],
            "sampling": "Current student, temperature=1, top_p=1, top_k=0, one rollout/input",
            "checkpoint_selection": "None; periodic checkpoints and final are all retained",
            "adaptation": "Multimodal four-field Q-Reflector; auxiliary SFT is an extension to OPSD",
            "gpu_started": False}
    write_json(args.output / "plan.json", plan)
    student, teacher = build_messages(rows[0], config["reference_fields"])
    write_json(args.output / "prompt_example.json", {
        "training_line": 1, "student_messages": student, "teacher_messages": teacher,
        "student_reference_is_not_an_input": True,
    })
    write_json(args.output / "status.json", {"status": "prepared", "gpu_started": False})
    print(json.dumps({"status": "prepared", "run_dir": str(args.output.resolve()),
                      "rows": len(rows), "optimizer_steps": steps, "gpu_started": False}))


def load_plan(run_dir):
    plan = read_json(run_dir / "plan.json")
    if plan["worker_sha256"] != digest(__file__):
        raise ValueError("Worker changed after preparation; prepare a new run directory")
    for name, expected in plan["pinned_files"].items():
        if digest(name) != expected:
            raise ValueError(f"Pinned model/processor file changed: {name}")
    rows, validation = validate_training_data(Path(plan["config"]["data"]))
    if validation != plan["data_validation"]:
        raise ValueError("Training data changed after preparation")
    return plan, rows


def cpu_environment():
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_FLAX="0",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")


def materialize_messages(messages, data, max_side):
    from PIL import Image

    result = copy.deepcopy(messages)
    for message in result:
        for item in message["content"]:
            if item["type"] != "image":
                continue
            with Image.open(data / item["image"]) as opened:
                picture = opened.convert("RGB")
            if max(picture.size) > max_side:
                scale = max_side / max(picture.size)
                picture = picture.resize((int(picture.width * scale), int(picture.height * scale)),
                                         Image.Resampling.LANCZOS)
            item["image"] = picture
    return result


def encode_messages(processor, messages, generation_prompt=True):
    return dict(processor.apply_chat_template(
        messages, add_generation_prompt=generation_prompt, tokenize=True,
        return_dict=True, return_tensors="pt", truncation=False))


def trim_at_eos(ids, eos_ids):
    """Include first EOS, exclude every token after it. Single unpadded response."""
    stops = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Expected one nonempty response")
    for i, token in enumerate(ids[0].tolist()):
        if token in stops:
            return ids[:, :i + 1]
    return ids


def encode_example(processor, row, config, eos_ids):
    import torch

    student, teacher = build_messages(row, config["reference_fields"])
    data = Path(config["data"])
    student = materialize_messages(student, data, config["image_max_side"])
    # Reuse the same image objects, in the same order, for the teacher.
    teacher[0]["content"] = student[0]["content"] + [teacher[0]["content"][-1]]
    student_batch = encode_messages(processor, student)
    teacher_batch = encode_messages(processor, teacher)
    full = encode_messages(processor, student + [row["messages"][1]], generation_prompt=False)
    prefix_length = student_batch["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :prefix_length], student_batch["input_ids"]):
        raise ValueError("Assistant target tokenization changed the student prompt prefix")
    reference_ids = trim_at_eos(full["input_ids"][:, prefix_length:], eos_ids)
    for key in ("pixel_values", "image_grid_thw"):
        if key not in student_batch or not torch.equal(student_batch[key], teacher_batch[key]):
            raise ValueError(f"Teacher/student image inputs differ: {key}")
        teacher_batch[key] = student_batch[key]
    limit = config["max_sequence_length"]
    lengths = {"student_prompt_tokens": prefix_length,
               "teacher_prompt_tokens": teacher_batch["input_ids"].shape[1],
               "reference_tokens": reference_ids.shape[1]}
    if (max(lengths["student_prompt_tokens"], lengths["teacher_prompt_tokens"])
            + config["max_new_tokens"] > limit
            or prefix_length + reference_ids.shape[1] > limit):
        raise ValueError(f"Sequence would exceed {limit}; silent truncation is forbidden: {lengths}")
    return student_batch, teacher_batch, reference_ids, lengths


def eos_from_processor(processor):
    token_id = processor.tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    return [token_id]


def audit(args):
    cpu_environment()
    plan, rows = load_plan(args.run_dir)
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(plan["config"]["adapter"], local_files_only=True)
    entries = []
    for i, row in enumerate(rows, 1):
        _, _, _, lengths = encode_example(processor, row, plan["config"], eos_from_processor(processor))
        entries.append({"training_line": i, **lengths})
        if i % 100 == 0:
            print(f"CPU token audit {i}/{len(rows)}", flush=True)
    report = {"status": "passed", "gpu_started": False, "rows": len(rows),
              "plan_sha256": digest(args.run_dir / "plan.json"), "samples": entries,
              "image_order_equal": True, "student_reference_leakage": False,
              "target_prefix_exact": True, "truncated_samples": 0}
    write_json(args.run_dir / "token_audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}))


def append_response(prompt, response_ids):
    import torch

    if response_ids.ndim != 2 or response_ids.shape[0] != 1 or response_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty response")
    if prompt["input_ids"].shape[0] != 1 or not bool(prompt["attention_mask"].all()):
        raise ValueError("Only unpadded single-example batches are supported")
    result = {k: v for k, v in prompt.items()
              if k not in ("position_ids", "cache_position", "past_key_values", "token_type_ids")}
    result["input_ids"] = torch.cat((prompt["input_ids"], response_ids), dim=1)
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    return result


def response_logits(model, prompt, response_ids):
    """Logit at prompt_length-1 predicts response[0], including first-token loss."""
    count = response_ids.shape[1]
    output = model(**append_response(prompt, response_ids), use_cache=False,
                   logits_to_keep=count + 1)
    if output.logits.shape[1] != count + 1:
        raise ValueError("Model did not honor logits_to_keep; response alignment is unsafe")
    return output.logits[:, :-1, :]


def full_vocab_forward_kl(student_logits, teacher_logits, pointwise_clip=0.05, chunk_tokens=16):
    """Teacher stop-gradient; clip vocabulary contributions BEFORE summation.

    A clipped forward-KL sum can be negative. Zero clipping disables clipping.
    Chunk checkpointing reduces fp32 softmax activation storage, without top-k bias.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Teacher/student response logits must align exactly")
    if student_logits.shape[1] == 0 or chunk_tokens <= 0 or pointwise_clip < 0:
        raise ValueError("Invalid loss dimensions or clipping")
    teacher_logits = teacher_logits.detach()

    def chunk_loss(student, teacher):
        log_student = student.float().log_softmax(-1)
        log_teacher = teacher.float().log_softmax(-1)
        contributions = log_teacher.exp() * (log_teacher - log_student)
        if pointwise_clip:
            contributions = contributions.clamp(max=pointwise_clip)
        return contributions.sum()

    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, student_logits.shape[1], chunk_tokens):
        s = student_logits[:, start:start + chunk_tokens]
        t = teacher_logits[:, start:start + chunk_tokens]
        if s.requires_grad:
            total = total + checkpoint(chunk_loss, s, t, use_reentrant=False)
        else:
            total = total + chunk_loss(s, t)
    return total / (student_logits.shape[0] * student_logits.shape[1])


def reference_cross_entropy(logits, targets, chunk_tokens=16):
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if logits.shape[:2] != targets.shape:
        raise ValueError("SFT response logits and target IDs do not align")

    def chunk_loss(scores, ids):
        return functional.cross_entropy(scores.float().reshape(-1, scores.shape[-1]),
                                        ids.reshape(-1), reduction="sum")

    loss = logits.new_zeros(())
    for start in range(0, targets.shape[1], chunk_tokens):
        loss = loss + checkpoint(chunk_loss, logits[:, start:start + chunk_tokens],
                                 targets[:, start:start + chunk_tokens], use_reentrant=False)
    return loss / targets.numel()


def select_adapter(model, name):
    """PEFT set_adapter can re-enable gradients: explicitly freeze the teacher."""
    if name not in (STUDENT, TEACHER):
        raise ValueError("Unknown OPSD adapter")
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        is_student = "lora_" in parameter_name and f".{STUDENT}." in parameter_name
        parameter.requires_grad_(name == STUDENT and is_student)


def fingerprint_adapter(model, name):
    import torch
    from peft import get_peft_model_state_dict

    h = hashlib.sha256()
    for key, tensor in sorted(get_peft_model_state_dict(model, adapter_name=name).items()):
        h.update(key.encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_loaded_adapter(model, source, name):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    loaded = get_peft_model_state_dict(model, adapter_name=name)
    with safe_open(str(source / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        if set(loaded) != set(saved.keys()):
            raise ValueError(f"Adapter key mismatch: {name}")
        for key, value in loaded.items():
            if not torch.equal(value.detach().cpu(), saved.get_tensor(key).to(value.dtype)):
                raise ValueError(f"Adapter tensor mismatch: {name}/{key}")
    return len(loaded)


def epoch_batches(count, accumulation, seed, epoch):
    indices = list(range(count))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[i:i + accumulation] for i in range(0, count, accumulation)]


def learning_rate_at(step, total, maximum, warmup_ratio):
    warmup = math.ceil(total * warmup_ratio)
    if warmup and step <= warmup:
        return maximum * step / warmup
    progress = (step - warmup - 1) / max(1, total - warmup)
    return maximum * 0.5 * (1 + math.cos(math.pi * progress))


def gpu_guard(torch, reserve_gib):
    free, _ = torch.cuda.mem_get_info(0)
    if free < reserve_gib * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.2f} GiB free; stopping this run for headroom")


def save_checkpoint(model, processor, optimizer, run_dir, state, teacher_hash, final=False):
    import torch

    if fingerprint_adapter(model, TEACHER) != teacher_hash:
        raise RuntimeError("Frozen teacher changed")
    destination = run_dir / ("final" if final else f"checkpoint-{state['step']}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temporary = run_dir / f".saving-{uuid.uuid4().hex}"
    temporary.mkdir()
    select_adapter(model, STUDENT)
    model.save_pretrained(str(temporary), selected_adapters=[STUDENT], safe_serialization=True)
    processor.save_pretrained(str(temporary))
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()},
               temporary / "optimizer_rng.pt")
    write_json(temporary / "opsd_state.json", {**state, "teacher_sha256": teacher_hash,
               "plan_sha256": digest(run_dir / "plan.json"), "complete": True})
    temporary.rename(destination)


def train(args):
    """Explicit foreground launch only; no process management or auto evaluation."""
    import fcntl

    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        raise ValueError("Supply one full GPU UUID; numeric/default GPU selection is forbidden")
    if args.memory_cap_gib <= 0 or args.reserve_free_gib < 0:
        raise ValueError("Invalid GPU memory limits")
    plan, rows = load_plan(args.run_dir)
    audit_report = read_json(args.run_dir / "token_audit.json")
    if (audit_report.get("status") != "passed" or audit_report["rows"] != len(rows)
            or audit_report["plan_sha256"] != digest(args.run_dir / "plan.json")):
        raise ValueError("Run the complete CPU token audit for this exact plan first")
    if (args.run_dir / "final").exists():
        raise FileExistsError("This run already has a final checkpoint")
    if not args.resume and (args.run_dir / "runtime.json").exists():
        raise ValueError("A launch already exists; resume a checkpoint or prepare a new run")
    resume_state = None
    if args.resume:
        resume_state = read_json(args.resume / "opsd_state.json")
        if not resume_state.get("complete") or resume_state["plan_sha256"] != digest(args.run_dir / "plan.json"):
            raise ValueError("Resume checkpoint does not belong to this plan")
    with (args.run_dir / ".train.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                          PYTORCH_ALLOC_CONF="expandable_segments:True")
        try:
            _train_locked(args, plan, rows, resume_state)
        except BaseException as error:
            previous = read_json(args.run_dir / "status.json")
            write_json(args.run_dir / "status.json", {**previous, "status": "failed",
                       "error": f"{type(error).__name__}: {error}", "pid": os.getpid()})
            raise


def _train_locked(args, plan, rows, resume_state):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    config = plan["config"]
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected GPU must be visible")
    free, capacity = torch.cuda.mem_get_info(0)
    if free < (args.memory_cap_gib + args.reserve_free_gib) * 1024**3:
        raise RuntimeError("Insufficient free GPU memory for cap plus reserve; nothing unloaded")
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 1024**3 / capacity, 0)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    attempt = uuid.uuid4().hex
    write_json(args.run_dir / "status.json", {"status": "loading", "pid": os.getpid(),
               "gpu_uuid": args.gpu_uuid, "attempt_id": attempt})
    write_json(args.run_dir / "runtime.json", {"gpu_uuid": args.gpu_uuid,
               "memory_cap_gib": args.memory_cap_gib, "reserve_free_gib": args.reserve_free_gib,
               "attempt_id": attempt, "resume": str(args.resume) if args.resume else None,
               "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}})
    processor = AutoProcessor.from_pretrained(config["adapter"], local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        config["base"], dtype=torch.bfloat16, device_map={"": "cuda:0"},
        local_files_only=True, attn_implementation="sdpa")
    if getattr(base.config, "quantization_config", None):
        raise ValueError("OPSD entry point requires the unquantized BF16 base")
    student_source = args.resume or Path(config["adapter"])
    model = PeftModel.from_pretrained(base, str(student_source), adapter_name=STUDENT,
                                     is_trainable=True)
    model.load_adapter(config["adapter"], adapter_name=TEACHER, is_trainable=False)
    verified = {STUDENT: verify_loaded_adapter(model, student_source, STUDENT),
                TEACHER: verify_loaded_adapter(model, Path(config["adapter"]), TEACHER)}
    teacher_hash = fingerprint_adapter(model, TEACHER)
    if resume_state and resume_state["teacher_sha256"] != teacher_hash:
        raise ValueError("Teacher changed since saved checkpoint")
    select_adapter(model, STUDENT)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable student adapter parameters")
    write_json(args.run_dir / "adapter_validation.json", {
        "verified_tensors": verified, "frozen_teacher_sha256": teacher_hash,
        "student_trainable_parameters": sum(p.numel() for p in parameters),
        "shared_frozen_base": True})
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
    state = {"step": 0, "next_epoch": 0, "next_batch": 0}
    if resume_state:
        payload = torch.load(args.resume / "optimizer_rng.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        state = {key: resume_state[key] for key in state}
        del payload
    eos_ids = eos_from_processor(processor)
    generation_eos = model.generation_config.eos_token_id
    if generation_eos is not None:
        eos_ids = sorted(set(eos_ids + ([generation_eos] if isinstance(generation_eos, int) else generation_eos)))
    suppressed = sorted({getattr(base.config, name) for name in
                         ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
                         if isinstance(getattr(base.config, name, None), int)})
    started = time.monotonic()
    with (args.run_dir / "metrics.jsonl").open("a") as metrics_file, \
            (args.run_dir / "rollouts.jsonl").open("a") as rollout_file:
        for epoch in range(state["next_epoch"], config["epochs"]):
            batches = epoch_batches(len(rows), config["gradient_accumulation"], config["seed"], epoch)
            first_batch = state["next_batch"] if epoch == state["next_epoch"] else 0
            for batch_index in range(first_batch, len(batches)):
                if state["step"] >= plan["optimizer_steps"]:
                    break
                indices = batches[batch_index]
                optimizer.zero_grad(set_to_none=True)
                opsd_sum = sft_sum = 0.0
                correct = invalid = limited = generated_tokens = 0
                for index in indices:
                    gpu_guard(torch, args.reserve_free_gib)
                    student, teacher, reference_ids, lengths = encode_example(
                        processor, rows[index], config, eos_ids)
                    student = {k: v.to("cuda:0") for k, v in student.items()}
                    teacher = {k: v.to("cuda:0") for k, v in teacher.items()}
                    reference_ids = reference_ids.to("cuda:0")
                    select_adapter(model, STUDENT)
                    model.eval()
                    with torch.no_grad():
                        generated = model.generate(
                            **student, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                            num_beams=1, max_new_tokens=config["max_new_tokens"], use_cache=True,
                            repetition_penalty=1.0, eos_token_id=eos_ids,
                            pad_token_id=processor.tokenizer.pad_token_id or eos_ids[0],
                            suppress_tokens=suppressed)
                        response = trim_at_eos(generated[:, student["input_ids"].shape[1]:], eos_ids)
                        if any(token in suppressed for token in response[0].tolist()):
                            raise ValueError("Student generated a visual placeholder token")
                        select_adapter(model, TEACHER)
                        teacher_scores = response_logits(model, teacher, response)
                    select_adapter(model, STUDENT)
                    model.train()
                    student_scores = response_logits(model, student, response)
                    loss = full_vocab_forward_kl(student_scores, teacher_scores,
                        config["pointwise_clip"], config["loss_chunk_tokens"])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite OPSD loss")
                    opsd_sum += loss.detach().item()
                    # Normalize by actual batch size, including the final partial batch.
                    (loss / len(indices)).backward()
                    del student_scores, teacher_scores, loss
                    if config["sft_weight"]:
                        reference_scores = response_logits(model, student, reference_ids)
                        sft = reference_cross_entropy(reference_scores, reference_ids, config["loss_chunk_tokens"])
                        if not bool(torch.isfinite(sft)):
                            raise FloatingPointError("Non-finite SFT loss")
                        sft_sum += sft.detach().item()
                        (config["sft_weight"] * sft / len(indices)).backward()
                        del reference_scores, sft
                    decoded = processor.decode(response[0], skip_special_tokens=True).strip()
                    try:
                        verdict = json.loads(decoded).get("answer")
                    except (ValueError, AttributeError):
                        verdict = None
                    expected = target_of(rows[index])["answer"]
                    correct += verdict == expected
                    invalid += verdict not in ("Yes", "No")
                    at_limit = response.shape[1] == config["max_new_tokens"] and response[0, -1].item() not in eos_ids
                    limited += at_limit
                    generated_tokens += response.shape[1]
                    rollout_file.write(json.dumps({"attempt_id": attempt, "update": state["step"] + 1,
                        "training_line": index + 1, "expected": expected, "predicted": verdict,
                        "response": decoded, "token_ids": response[0].tolist(),
                        "generation_limit": at_limit, **lengths}, ensure_ascii=False) + "\n")
                    del student, teacher, reference_ids, response, generated
                gpu_guard(torch, args.reserve_free_gib)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if any(p.grad is not None for n, p in model.named_parameters() if f".{TEACHER}." in n):
                    raise RuntimeError("Teacher received gradients")
                step = state["step"] + 1
                lr = learning_rate_at(step, plan["optimizer_steps"], config["learning_rate"], config["warmup_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                at_epoch_end = batch_index + 1 == len(batches)
                state = {"step": step, "next_epoch": epoch + 1 if at_epoch_end else epoch,
                         "next_batch": 0 if at_epoch_end else batch_index + 1}
                metrics = {**state, "attempt_id": attempt, "epoch": epoch + 1,
                    "batch_examples": len(indices), "opsd_loss": opsd_sum / len(indices),
                    "sft_loss": sft_sum / len(indices), "sft_weight": config["sft_weight"],
                    "training_rollout_accuracy": correct / len(indices), "invalid_outputs": invalid,
                    "generation_limit_count": limited, "generated_tokens": generated_tokens,
                    "learning_rate": lr, "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                rollout_file.flush()
                write_json(args.run_dir / "status.json", {"status": "training", "pid": os.getpid(),
                           "gpu_uuid": args.gpu_uuid, "total_steps": plan["optimizer_steps"], **metrics})
                print(json.dumps(metrics), flush=True)
                if step % config["save_every"] == 0 or at_epoch_end:
                    save_checkpoint(model, processor, optimizer, args.run_dir, state, teacher_hash)
            if state["step"] >= plan["optimizer_steps"]:
                break
    save_checkpoint(model, processor, optimizer, args.run_dir, state, teacher_hash, final=True)
    write_json(args.run_dir / "status.json", {"status": "complete", "pid": os.getpid(),
               "gpu_uuid": args.gpu_uuid, **state, "teacher_unchanged": True,
               "elapsed_seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="Validate train data and write a plan; no GPU/model")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--adapter", type=Path, required=True, help="Explicit initial SFT checkpoint")
    prep.add_argument("--data", type=Path, default=DATA)
    prep.add_argument("--base", type=Path, default=BASE)
    prep.add_argument("--reference-fields", choices=("full", "answer"), default="full")
    prep.add_argument("--epochs", type=int, default=1)
    prep.add_argument("--max-steps", type=int, default=0, help="0 means use all requested epochs")
    prep.add_argument("--gradient-accumulation", type=int, default=8)
    prep.add_argument("--learning-rate", type=float, default=5e-6)
    prep.add_argument("--warmup-ratio", type=float, default=0.1)
    prep.add_argument("--sft-weight", type=float, default=0.1)
    prep.add_argument("--pointwise-clip", type=float, default=0.05, help="0 disables pointwise clipping")
    prep.add_argument("--max-new-tokens", type=int, default=512)
    prep.add_argument("--max-sequence-length", type=int, default=8192)
    prep.add_argument("--image-max-side", type=int, default=512)
    prep.add_argument("--loss-chunk-tokens", type=int, default=16)
    prep.add_argument("--save-every", type=int, default=25)
    prep.add_argument("--seed", type=int, default=42)
    check = sub.add_parser("audit", help="Tokenize all training inputs/targets on CPU; no weights")
    check.add_argument("--run-dir", type=Path, required=True)
    launch = sub.add_parser("train", help="Explicit GPU training; never launches automatically")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--gpu-uuid", required=True)
    launch.add_argument("--memory-cap-gib", type=float, default=40)
    launch.add_argument("--reserve-free-gib", type=float, default=8)
    launch.add_argument("--resume", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "audit": audit, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
