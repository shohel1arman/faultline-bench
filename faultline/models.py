"""Hugging Face backend for vision-language models.

One code path for every model: render the chat template to a string with image
placeholders, then let the processor tokenize text and image together. torch and
transformers are imported lazily so the rest of the package works without them.
"""

import math

from .prompts import LETTERS


def mps_version_problem(torch_version, macos_version):
    """Return a message if this torch build is known to fail on this macOS, else None.

    torch 2.15 calls Apple GPU (MPS) functions missing from macOS 14, which aborts the
    process with "unrecognized selector ... arrayViewWithShape:strides:".
    """
    def major_minor(v):
        parts = (v.split("+")[0].split(".") + ["0", "0"])[:2]
        return tuple(int("".join(ch for ch in x if ch.isdigit()) or 0) for x in parts)

    if not macos_version:
        return None
    if major_minor(torch_version) >= (2, 15) and major_minor(macos_version)[0] < 15:
        return (f"torch {torch_version} needs macOS 15 or newer on the Apple GPU, but this Mac "
                f"runs macOS {macos_version}. Run `uv sync --extra eval --group dev` to install "
                "the torch version pinned for macOS in pyproject.toml.")
    return None


def load_image(path, max_side):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BICUBIC)
    return img


def _messages(turns):
    msgs = []
    for t in turns:
        if t["role"] == "assistant":
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
            continue
        content = [{"type": "image"}] if t.get("image") is not None else []
        content.append({"type": "text", "text": t["text"]})
        msgs.append({"role": "user", "content": content})
    return msgs


class HFBackend:
    def __init__(self, hf_id, device="cuda:0", quant="nf4", dtype="float16"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.device = device
        torch_dtype = getattr(torch, dtype)
        self.dtype = torch_dtype
        on_cuda = device.startswith("cuda")
        if quant != "none" and not on_cuda:
            raise ValueError("4-bit and 8-bit loading need a CUDA GPU; use quant = \"none\" on Apple GPUs")
        kwargs = {"dtype": torch_dtype}
        if on_cuda:
            kwargs["device_map"] = {"": device}
        if quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant != "none":
            raise ValueError(f"unknown quant {quant}")

        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
        if not on_cuda:
            self.model.to(device)
        self.model.eval()
        self.tok = self.processor.tokenizer
        self.letter_ids = {L: self._letter_token_ids(L) for L in LETTERS}

    def _letter_token_ids(self, letter):
        ids = set()
        for variant in (letter, " " + letter):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _inputs(self, turns):
        prompt = self.processor.apply_chat_template(
            _messages(turns), add_generation_prompt=True, tokenize=False
        )
        images = [t["image"] for t in turns if t.get("image") is not None]
        kw = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kw["images"] = images
        inputs = self.processor(**kw)
        # floating tensors (pixel values) are cast to the compute dtype, ids stay integer
        return inputs.to(self.device, dtype=self.dtype)

    def generate(self, turns, max_new_tokens):
        inputs = self._inputs(turns)
        n_in = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tok.decode(out[0, n_in:], skip_special_tokens=True)
        return {"output": text.strip(), "n_in": int(n_in), "n_out": int(out.shape[1] - n_in)}

    def letter_probs(self, turns):
        """Softmax over the four option letters at the first answer position.

        Also returns the share of total next-token probability that falls on the
        letters, which shows whether the model is answering in the expected format.
        """
        inputs = self._inputs(turns)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        if not self.torch.isfinite(logits).all():
            return {"error": "non-finite logits"}
        logp = self.torch.log_softmax(logits, dim=-1)
        per_letter = {}
        for L, ids in self.letter_ids.items():
            per_letter[L] = self.torch.logsumexp(logp[ids], dim=0).item()
        mass = sum(math.exp(v) for v in per_letter.values())
        z = max(per_letter.values())
        exp = {L: math.exp(v - z) for L, v in per_letter.items()}
        tot = sum(exp.values())
        probs = {L: exp[L] / tot for L in LETTERS}
        return {"probs": probs, "letter_mass": mass, "n_in": int(inputs["input_ids"].shape[1])}

    def cleanup(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        elif self.device == "mps":
            self.torch.mps.empty_cache()

    def memory_gib(self):
        if self.device.startswith("cuda"):
            return self.torch.cuda.memory_allocated(self.device) / 2**30
        if self.device == "mps":
            return self.torch.mps.current_allocated_memory() / 2**30
        return None
