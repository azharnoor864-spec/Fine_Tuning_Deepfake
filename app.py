"""
Day 32 — AI Safety / Deepfake Misinformation Domain Model — Gradio Demo
Loads the merged (Day 31) Qwen2.5-1.5B model in 4-bit NF4 (bitsandbytes) and
runs inference on the GPU allocated by HuggingFace Spaces' ZeroGPU free tier.

Model is lazy-loaded on the first generation call (inside the @spaces.GPU
context, when a real CUDA device is actually attached) and then cached for
subsequent calls within the same worker.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import spaces

# ===== Config =====
REPO_ID = "nooruiit-864/qwen2.5-1.5b-base-ai-safety-domain-lora"

# Tokenizer has no GPU dependency — safe to load at startup
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

_model = None  # populated on first GPU-context call, then cached


def _get_model():
    """
    Lazy-load the merged model in 4-bit NF4 (bitsandbytes) the first time a
    generation request runs. bitsandbytes 4-bit quantization requires a CUDA
    device to be present at load time — on ZeroGPU Spaces, a real GPU is only
    attached while a @spaces.GPU-decorated function is executing, so loading
    must happen here rather than at module import time.
    """
    global _model
    if _model is None:
        print("First request: loading model onto GPU with 4-bit NF4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            REPO_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _model.generation_config.pad_token_id = tokenizer.eos_token_id
        print("Model loaded on GPU. Ready for inference.")
    return _model


@spaces.GPU(duration=90)
def generate_response(message, history, max_tokens, temperature):
    """
    Plain continuation-style generation (this is a domain-adapted base model,
    NOT an instruction-tuned chat model — it completes text rather than
    following chat turns).

    Duration is set to 90s to comfortably cover the first (cold-start) call,
    which includes downloading + 4-bit quantizing a ~3GB model in addition to
    generation. Subsequent calls reuse the cached model and run much faster,
    consistent with the Day 31 bitsandbytes benchmark (~14.9 tokens/sec on a
    T4-class GPU).
    """
    if not message.strip():
        return "Please enter a prompt."

    model = _get_model()
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text[len(message):].strip()


EXAMPLE_PROMPTS = [
    "Deepfake technology has made it increasingly difficult to",
    "Content authenticity standards such as C2PA are designed to",
    "The biggest risk of generative AI misuse is",
    "Governments are responding to synthetic media by",
    "AI alignment research focuses on",
]

with gr.Blocks(title="Deepfake & AI Safety Domain Model") as demo:
    gr.Markdown(
        """
        # 🛡️ AI Safety / Deepfake Misinformation — Domain-Adapted Qwen2.5-1.5B

        A QLoRA domain-adapted checkpoint of **Qwen2.5-1.5B (base)**, fine-tuned on a
        700-passage corpus covering deepfakes, synthetic media, and AI-related
        misinformation. Merged (Day 31), running in **4-bit NF4 on GPU** via
        HuggingFace Spaces' ZeroGPU.

        ⚠️ **This is a continuation model, not a chat assistant** — enter a sentence
        prefix and the model will continue it in the domain's style. The first
        request after the Space wakes up may take ~20–40s (model load); later
        requests are much faster. See the
        [model card](https://huggingface.co/nooruiit-864/qwen2.5-1.5b-base-ai-safety-domain-lora)
        for training details, benchmarks, and known limitations (including a
        documented safety caveat).
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Prompt (sentence prefix)",
                placeholder="e.g. Deepfake technology has made it increasingly difficult to",
                lines=3,
            )
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=20, maximum=200, value=80, step=10, label="Max new tokens"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.2, value=0.7, step=0.1, label="Temperature"
                )
            generate_btn = gr.Button("Generate", variant="primary")
            output_box = gr.Textbox(label="Model continuation", lines=6)

        with gr.Column(scale=1):
            gr.Markdown("**Try an example prompt:**")
            example_buttons = [gr.Button(p, size="sm") for p in EXAMPLE_PROMPTS]

    generate_btn.click(
        fn=lambda p, mt, t: generate_response(p, None, mt, t),
        inputs=[prompt_input, max_tokens_slider, temperature_slider],
        outputs=output_box,
    )

    for btn, prompt_text in zip(example_buttons, EXAMPLE_PROMPTS):
        btn.click(fn=lambda pt=prompt_text: pt, outputs=prompt_input)

    gr.Markdown(
        """
        ---
        **Base model:** Qwen2.5-1.5B (non-instruct) · **Method:** QLoRA (4-bit NF4) →
        merged → 4-bit NF4 (bitsandbytes, GPU) · **Training corpus:** 700
        Wikipedia-sourced passages, AI safety / deepfake / misinformation domain ·
        Built as part of the Planet Beyond AI Engineer Internship (Day 27–32).
        """
    )

if __name__ == "__main__":
    demo.launch()
