"""Private JSONL worker. Run by GUIReflectionReflector with the isolated Python.

All model imports are inside main; importing/help does not initialize CUDA.
The official wrapper owns InternVL chat serialization and image preprocessing.
"""

import argparse
import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path


def install_token_counter(model):
    """Measure the official InternVL wrapper without changing generation.

    Its generate() receives expanded image/text input_ids and returns only the
    continuation: the underlying language model is called with inputs_embeds.
    """
    counts = {}
    generate = model.generate

    def measured_generate(*args, **kwargs):
        generated = generate(*args, **kwargs)
        sequences = getattr(generated, "sequences", generated)
        counts.update(
            prompt_tokens=int(kwargs["input_ids"].shape[-1]),
            completion_tokens=int(sequences.shape[-1]),
            source="model_token_ids",
        )
        return generated

    model.generate = measured_generate
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--official-code", required=True)
    parser.add_argument("--memory-cap-gib", type=float, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # Keep stdout exclusively for the protocol, including around native imports.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(message):
        protocol.write(json.dumps(message, ensure_ascii=False) + "\n")
        protocol.flush()

    request_id = None
    try:
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
            raise RuntimeError("Worker requires an explicitly selected physical GPU UUID")
        os.nice(10)
        import torch
        from PIL import Image

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible GPU")
        free, capacity = torch.cuda.mem_get_info(0)
        cap = args.memory_cap_gib * 2**30
        if not 0 < cap <= capacity or free < cap + 8 * 2**30:
            raise RuntimeError(
                f"Insufficient GPU headroom: {free / 2**30:.1f} GiB free; "
                f"need {args.memory_cap_gib + 8:g} GiB before loading"
            )
        torch.cuda.set_per_process_memory_fraction(cap / capacity, 0)
        sys.path.insert(0, str(Path(args.official_code) / "internvl_chat"))
        from eval.infer import Model

        model = Model(args.model_path, max_num=6, auto=False)
        token_usage = install_token_counter(model.model)
        send({"type": "ready"})
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request["id"]
            if len(request["images"]) != 2 or not isinstance(request["prompt"], str):
                raise ValueError("Expected a prompt and an ordered before/after image pair")
            if torch.cuda.mem_get_info(0)[0] < 8 * 2**30:
                torch.cuda.empty_cache()
                if torch.cuda.mem_get_info(0)[0] < 8 * 2**30:
                    raise RuntimeError("GPU free headroom dropped below 8 GiB; stopping this worker")
            images = []
            for encoded in request["images"]:
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    images.append(image.convert("RGB"))
            with torch.inference_mode():
                token_usage.clear()
                raw = model(request["prompt"], images, {
                    "max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1,
                })
            torch.cuda.synchronize()
            send({"type": "result", "id": request_id, "raw_output": raw,
                  "token_usage": token_usage or None})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"type": "error", "id": request_id, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
