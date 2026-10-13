"""CPU-only tests of loss math, leakage, alignment, frozen adapters and restart state."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Set visibility before importing torch/transformers; never load production weights.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import qreflector_opsd as opsd  # noqa: E402 -- isolate CUDA before ML imports
import torch  # noqa: E402

torch.set_num_threads(2)


def example_row():
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Task: Find attractions. Action: tap search."},
            {"type": "image", "image": "before.png"},
            {"type": "image", "image": "after.png"}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps({
            "precondition": "REFERENCE_PRECONDITION_ONLY",
            "postcondition": "REFERENCE_POSTCONDITION_ONLY",
            "reason": "REFERENCE_REASON_ONLY", "answer": "Yes"})}]}]}


class PromptTests(unittest.TestCase):
    def test_reference_only_in_teacher_and_original_unmodified(self):
        row = example_row()
        original = copy.deepcopy(row)
        student, teacher = opsd.build_messages(row)
        self.assertEqual(student, [original["messages"][0]])
        self.assertNotIn("REFERENCE_REASON_ONLY", json.dumps(student))
        self.assertIn("REFERENCE_REASON_ONLY", json.dumps(teacher))
        self.assertEqual(row, original)
        self.assertEqual(teacher[0]["content"][:-1], student[0]["content"])
        teacher[0]["content"][0]["text"] = "Mutated teacher"
        self.assertEqual(row, original)
        self.assertEqual(student, [original["messages"][0]])

    def test_answer_only_privilege_does_not_include_reference_rationale(self):
        student, teacher = opsd.build_messages(example_row(), "answer")
        self.assertNotIn("REFERENCE_REASON_ONLY", json.dumps(teacher))
        self.assertIn('"answer": "Yes"', teacher[0]["content"][-1]["text"])
        self.assertNotIn(opsd.REFERENCE_MARKER, json.dumps(student))

    def test_reference_in_student_and_invalid_reference_rejected(self):
        row = example_row()
        row["messages"][0]["content"][0]["text"] += opsd.REFERENCE_MARKER
        with self.assertRaises(ValueError):
            opsd.build_messages(row)
        row = example_row()
        row["messages"][1]["content"][0]["text"] = '{"answer": "Maybe"}'
        with self.assertRaises(ValueError):
            opsd.build_messages(row)


class LossTests(unittest.TestCase):
    def test_chunked_loss_and_gradient_equal_dense_kl_and_teacher_gets_no_gradient(self):
        torch.manual_seed(17)
        student = torch.randn(1, 7, 19, requires_grad=True)
        teacher = torch.randn(1, 7, 19, requires_grad=True)
        actual = opsd.full_vocab_forward_kl(student, teacher, pointwise_clip=0, chunk_tokens=2)
        actual.backward()
        actual_grad = student.grad.clone()
        self.assertIsNone(teacher.grad)
        reference = student.detach().clone().requires_grad_()
        tlog = teacher.detach().log_softmax(-1)
        expected = (tlog.exp() * (tlog - reference.log_softmax(-1))).sum(-1).mean()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, reference.grad)

    def test_clipping_is_per_vocabulary_entry_not_after_sum(self):
        student = torch.tensor([[[2., -4., -1., 0.]]], requires_grad=True)
        teacher = torch.tensor([[[-1., 4., 0., 0.]]])
        tlog = teacher.log_softmax(-1)
        terms = tlog.exp() * (tlog - student.log_softmax(-1))
        expected = terms.clamp(max=0.05).sum(-1).mean()
        actual = opsd.full_vocab_forward_kl(student, teacher, pointwise_clip=0.05)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.isclose(actual, terms.sum(-1).clamp(max=0.05).mean()))

    def test_identical_distributions_have_zero_kl_and_gradient(self):
        scores = torch.randn(1, 3, 11, requires_grad=True)
        loss = opsd.full_vocab_forward_kl(scores, scores.detach(), pointwise_clip=0)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0, places=6)
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=1e-7, rtol=0)

    def test_sft_loss_includes_first_response_token_and_eos(self):
        scores = torch.randn(1, 5, 13, requires_grad=True)
        target = torch.tensor([[4, 7, 3, 9, 2]])
        loss = opsd.reference_cross_entropy(scores, target, chunk_tokens=2)
        expected = torch.nn.functional.cross_entropy(scores.reshape(-1, 13), target.reshape(-1))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(float(scores.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(scores.grad[0, -1].abs().sum()), 0)


class AlignmentTests(unittest.TestCase):
    def test_logits_predict_first_and_last_response_at_unequal_prompt_lengths(self):
        class ToyModel:
            def __call__(self, input_ids, logits_to_keep, **kwargs):
                scores = torch.nn.functional.one_hot((input_ids + 1) % 32, 32).float()
                return SimpleNamespace(logits=scores[:, -logits_to_keep:])

        response = torch.tensor([[4, 5, 6]])
        for ids in [torch.tensor([[2, 3]]), torch.tensor([[19, 21, 24, 2, 3]])]:
            prompt = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            actual = opsd.response_logits(ToyModel(), prompt, response)
            torch.testing.assert_close(actual.argmax(-1), response)
            self.assertEqual(prompt["input_ids"].shape[1], ids.shape[1])

    def test_eos_included_but_padding_or_later_tokens_are_excluded(self):
        ids = torch.tensor([[5, 8, 2, 0, 0]])
        torch.testing.assert_close(opsd.trim_at_eos(ids, [2]), torch.tensor([[5, 8, 2]]))
        torch.testing.assert_close(opsd.trim_at_eos(ids[:, :2], [2]), ids[:, :2])
        with self.assertRaises(ValueError):
            opsd.trim_at_eos(ids[:, :0], [2])

    def test_partial_batches_and_resume_order(self):
        batches = opsd.epoch_batches(882, 8, 42, 0)
        self.assertEqual(len(batches), 111)
        self.assertEqual(len(batches[-1]), 2)
        self.assertEqual(sorted(sum(batches, [])), list(range(882)))
        self.assertEqual(batches[25:], opsd.epoch_batches(882, 8, 42, 0)[25:])
        self.assertNotEqual(batches, opsd.epoch_batches(882, 8, 42, 1))


class TinyMultimodalIntegrationTests(unittest.TestCase):
    def test_real_qwen_vl_forward_backward_frozen_teacher_and_save_reload(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        torch.manual_seed(13)
        config = Qwen3VLConfig(
            text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 1, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 8,
                         "max_position_embeddings": 128,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "patch_size": 2, "spatial_merge_size": 2,
                           "temporal_patch_size": 1, "out_hidden_size": 32,
                           "num_position_embeddings": 16, "deepstack_visual_indexes": []},
            image_token_id=120, video_token_id=121, vision_start_token_id=122,
            vision_end_token_id=123, eos_token_id=2, pad_token_id=0,
        )
        base = Qwen3VLForConditionalGeneration(config)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0,
                                               target_modules=["q_proj", "v_proj"],
                                               task_type="CAUSAL_LM"))
        ids = torch.tensor([[1, 4, 122, 120, 123, 122, 120, 123, 6]])
        student = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                   "pixel_values": torch.randn(8, 12),
                   "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
        teacher_ids = torch.cat((ids, torch.tensor([[70, 71, 72]])), dim=1)
        teacher = {**student, "input_ids": teacher_ids, "attention_mask": torch.ones_like(teacher_ids)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            model.save_pretrained(initial)
            model.load_adapter(initial, adapter_name=opsd.TEACHER, is_trainable=False)
            self.assertGreater(opsd.verify_loaded_adapter(model, initial, opsd.TEACHER), 0)
            before = opsd.fingerprint_adapter(model, opsd.TEACHER)
            opsd.select_adapter(model, opsd.STUDENT)
            student_before = opsd.fingerprint_adapter(model, opsd.STUDENT)
            model.eval()
            with torch.no_grad():
                generated = model.generate(**student, max_new_tokens=3, do_sample=True,
                                           top_k=0, top_p=1., temperature=1.,
                                           suppress_tokens=[120, 121, 122, 123], eos_token_id=2,
                                           pad_token_id=0)
                response = opsd.trim_at_eos(generated[:, ids.shape[1]:], [2])
                opsd.select_adapter(model, opsd.TEACHER)
                target_scores = opsd.response_logits(model, teacher, response)
            opsd.select_adapter(model, opsd.STUDENT)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=0.01)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model.train()
            logits = opsd.response_logits(model, student, response)
            loss = opsd.full_vocab_forward_kl(logits, target_scores, pointwise_clip=0)
            loss.backward()
            reference = torch.tensor([[10, 11, 2]])
            sft = opsd.reference_cross_entropy(opsd.response_logits(model, student, reference), reference)
            (sft * 0.1).backward()
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                                if f".{opsd.TEACHER}." in n))
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in params if p.grad is not None), 0)
            optimizer.step()
            self.assertEqual(before, opsd.fingerprint_adapter(model, opsd.TEACHER))
            self.assertNotEqual(student_before, opsd.fingerprint_adapter(model, opsd.STUDENT))
            opsd.write_json(root / "plan.json", {"fixture": True})

            class ProcessorStub:
                def save_pretrained(self, path):
                    (Path(path) / "processor_fixture.json").write_text("{}")

            state = {"step": 1, "next_epoch": 0, "next_batch": 1}
            opsd.save_checkpoint(model, ProcessorStub(), optimizer, root, state, before)
            saved = root / "checkpoint-1"
            self.assertTrue((saved / "adapter_model.safetensors").is_file())
            self.assertFalse((saved / opsd.TEACHER).exists())
            payload = torch.load(saved / "optimizer_rng.pt", weights_only=True)
            self.assertTrue(payload["optimizer"]["state"])
            reloaded = PeftModel.from_pretrained(Qwen3VLForConditionalGeneration(config), saved,
                                                is_trainable=True)
            self.assertEqual(opsd.fingerprint_adapter(reloaded, opsd.STUDENT),
                             opsd.fingerprint_adapter(model, opsd.STUDENT))
            restored_optimizer = torch.optim.AdamW(
                [p for p in reloaded.parameters() if p.requires_grad], lr=0.001)
            restored_optimizer.load_state_dict(payload["optimizer"])
            original_state = optimizer.state_dict()
            restored_state = restored_optimizer.state_dict()
            self.assertEqual(original_state["param_groups"], restored_state["param_groups"])
            for parameter_id, moments in original_state["state"].items():
                for name, value in moments.items():
                    torch.testing.assert_close(value, restored_state["state"][parameter_id][name])
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(payload["torch_rng"])
                expected_random = torch.rand(8)
                torch.set_rng_state(payload["torch_rng"])
                torch.testing.assert_close(torch.rand(8), expected_random)
            self.assertEqual(opsd.read_json(saved / "opsd_state.json")["next_batch"], 1)
            with self.assertRaises(FileExistsError):
                opsd.save_checkpoint(model, ProcessorStub(), optimizer, root, state, before)


if __name__ == "__main__":
    unittest.main()
