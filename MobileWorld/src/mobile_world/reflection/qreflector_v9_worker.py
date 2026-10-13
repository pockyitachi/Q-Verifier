"""Private BF16 worker; no model imports or CUDA access outside main()."""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


def process_identity(pid):
    """Read PID identity without signalling it; start ticks protect against PID reuse."""
    process = Path('/proc') / str(pid)
    stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'uid': process.stat().st_uid, 'state': stat[0],
            'pgid': os.getpgid(pid), 'start_ticks': stat[19]}


def check_selected_gpu(gpu_uuid, companion_manifest=None):
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu_uuid):
        raise ValueError("Worker requires one physical GPU UUID")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", gpu_uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=15).strip()
    if not companion_manifest:
        if occupied:
            raise RuntimeError("Selected GPU has existing compute processes; refusing to load")
        return
    record = json.loads(Path(companion_manifest).read_text())
    if record.get('gpu_uuid') != gpu_uuid:
        raise RuntimeError('Companion manifest selects a different GPU')
    parent_pid = record['pid']
    parent = process_identity(parent_pid)
    if (parent['uid'] != os.getuid() or parent['state'] == 'Z'
            or parent['pgid'] != parent_pid or parent['start_ticks'] != str(record['start_ticks'])):
        raise RuntimeError('Companion identity or ownership changed')
    for value in occupied.splitlines():
        identity = process_identity(int(value.strip()))
        if identity['uid'] != os.getuid() or identity['pgid'] != parent_pid:
            raise RuntimeError('Selected GPU contains an unrelated compute process; refusing to load')


def encode_request(processor, request):
    from PIL import Image

    if not isinstance(request["prompt"], str) or len(request["images"]) != 2:
        raise ValueError("Expected a prompt and two ordered screenshots")
    content = [{"type": "text", "text": request["prompt"]}]
    for encoded in request["images"]:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            picture = image.convert("RGB")
        if max(picture.size) > 1024:
            raise ValueError("Parent must resize images consistently with geometry metadata")
        content.append({"type": "image", "image": picture})
    return dict(processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", truncation=False))


def generate_response(model, processor, inputs, max_new_tokens):
    n = inputs["input_ids"].shape[-1]
    if n + max_new_tokens > 8192:
        raise ValueError("Input plus output budget exceeds 8192; refusing silent truncation")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                            temperature=None, top_p=None, top_k=None, repetition_penalty=1.)
    ids = output[0, n:]
    return {"raw_output": processor.decode(ids, skip_special_tokens=True).strip(),
            "token_usage": {"prompt_tokens": int(n), "completion_tokens": len(ids),
                            "source": "model_token_ids"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("adapter", "base", "processor", "generation-config"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--memory-cap-gib", type=float, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--companion-process-manifest", default=None)
    args = parser.parse_args()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(message):
        protocol.write(json.dumps(message, ensure_ascii=False) + "\n")
        protocol.flush()

    request_id = None
    try:
        # Read-only preflight before importing torch or creating any CUDA context.
        check_selected_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", ""), args.companion_process_manifest)
        os.nice(10)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.manual_seed(42)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one GPU must be visible")
        free, capacity = torch.cuda.mem_get_info(0)
        cap = args.memory_cap_gib * 2**30
        if not 0 < cap <= capacity or free < cap + 8 * 2**30:
            raise RuntimeError("Insufficient free GPU headroom for the configured cap plus 8 GiB reserve")
        torch.cuda.set_per_process_memory_fraction(cap / capacity, 0)
        processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            local_files_only=True, attn_implementation="sdpa")
        base.generation_config = GenerationConfig.from_pretrained(args.generation_config, local_files_only=True)
        dense = [p for name, p in base.named_parameters() if p.ndim == 2 and ".layers." in name]
        if getattr(base.config, "quantization_config", None) or not dense or any(p.dtype != torch.bfloat16 for p in dense):
            raise RuntimeError("Expected a nonquantized BF16 base")
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Inference worker unexpectedly contains trainable parameters")
        send({"type": "ready"})
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request["id"]
            if torch.cuda.mem_get_info(0)[0] < 8 * 2**30:
                raise RuntimeError("Free GPU headroom dropped below 8 GiB; stopping only this worker")
            inputs = encode_request(processor, request)
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            with torch.inference_mode():
                result = generate_response(model, processor, inputs, args.max_new_tokens)
            torch.cuda.synchronize()
            send({"type": "result", "id": request_id, **result})
            del inputs
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"type": "error", "id": request_id, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
