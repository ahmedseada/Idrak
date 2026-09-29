"""Records what Hugging Face transformers produces for a pretrained model (token ids, rendered chat templates,
next-token logits and greedy continuations), so Idrak's own loader can be checked against it:

    pip install torch transformers huggingface_hub
    python tools/pytorch/pretrained_reference.py --model Qwen/Qwen3-0.6B --out qwen3.json
    dotnet run -c Release --project samples/Idrak.Samples.Chat -- check qwen3.json --cuda

--model is a Hugging Face model id (downloaded to the local cache) or a local folder. The reference file names the
folder, so the check reads exactly the files transformers used. --adapter applies a PEFT LoRA adapter (for example one
written by the sample's finetune command; pip install peft); the check then loads the same adapter. Any Llama-style model works the same way, for
example Qwen/Qwen2.5-Coder-1.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct or mistralai/Mistral-7B-Instruct-v0.3.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--out", default="reference.json")
parser.add_argument("--generate", type=int, default=24, help="greedy tokens to record per prompt")
parser.add_argument("--cpu", action="store_true", help="run transformers on the CPU even when a GPU is available")
parser.add_argument("--adapter", help="a PEFT LoRA adapter folder to apply to the model")
args = parser.parse_args()

folder = args.model
if not os.path.isdir(folder):
    from huggingface_hub import snapshot_download
    folder = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])

device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
tokenizer = AutoTokenizer.from_pretrained(folder)
model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float32).to(device).eval()
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()

texts = [
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
    "Hello, world! It's 2026; numbers: 1234567, 3.14159, 0x1F.",
    "    indented\tline\r\nwindows line\n\n\nblank lines   trailing spaces   ",
    "Unicode: café naïve Ünïcödé — “quotes” 你好世界 こんにちは 안녕하세요 🙂👍🏽 ∑∫√",
    "class Foo<T> : IBar where T : struct { public int X => 42; } // C#",
    "SELECT id, name FROM users WHERE email LIKE '%@example.com' ORDER BY id DESC;",
]

# Conversations in Idrak's terms (ChatMessage): converted below exactly as JinjaChatTemplate converts them.
tools = [
    {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path"}}, "required": ["path"]}},
    {"name": "run_tests", "description": "Run the test suite and return its output.",
     "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}},
]
agent = [
    {"role": "system", "content": "You are a coding agent. Use the tools to inspect and fix the repository."},
    {"role": "user", "content": "The tests fail with ZeroDivisionError. Fix it."},
    {"role": "assistant", "content": "", "thinking": "I should look at the code first.",
     "tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}]},
    {"role": "tool", "content": "def mean(xs):\n    return sum(xs) / len(xs)\n", "tool_name": "read_file"},
]
chats = [
    {"messages": agent, "tools": tools, "think": None, "add_generation_prompt": True},
    {"messages": agent, "tools": tools, "think": False, "add_generation_prompt": True},
    {"messages": agent + [{"role": "assistant", "content": "Empty lists now return 0.", "thinking": "Guard against empty input."}],
     "tools": tools, "think": None, "add_generation_prompt": False},
    {"messages": [{"role": "user", "content": "Write a Python function that reverses a string."}], "tools": [], "think": None,
     "add_generation_prompt": True},
]


def to_hf(messages):
    result, pending, counter = [], [], 0
    for m in messages:
        d = {"role": m["role"], "content": m["content"]}
        if "thinking" in m:
            d["reasoning_content"] = m["thinking"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                call_id = f"call{counter:05d}"
                counter += 1
                pending.append(call_id)
                calls.append({"type": "function", "id": call_id, "function": {"name": c["name"], "arguments": c["arguments"]}})
            d["tool_calls"] = calls
        if "tool_name" in m:
            d["name"] = m["tool_name"]
        if m["role"] == "tool" and pending:
            d["tool_call_id"] = pending.pop(0)
        result.append(d)
    return result


def render(chat):
    hf_tools = [{"type": "function", "function": t} for t in chat["tools"]] or None
    extra = {} if chat["think"] is None else {"enable_thinking": chat["think"]}
    try:
        return tokenizer.apply_chat_template(to_hf(chat["messages"]), tools=hf_tools, add_generation_prompt=chat["add_generation_prompt"],
                                             tokenize=False, **extra)
    except Exception as e:  # the model's template may reject a conversation (for example a system message)
        return "ERROR: " + str(e)


for chat in chats:
    chat["rendered"] = render(chat) if tokenizer.chat_template else "ERROR: no chat template"

prompts = [texts[0][:40], texts[4]]
if not chats[0]["rendered"].startswith("ERROR"):
    prompts.append(chats[0]["rendered"])
if not chats[3]["rendered"].startswith("ERROR"):
    prompts.append(chats[3]["rendered"])

runs = []
with torch.no_grad():
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids).logits[0, -1].float().cpu()
        generated = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=args.generate,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)[0, len(ids):].tolist()
        runs.append({"prompt": prompt, "ids": ids, "logits": [round(float(x), 5) for x in logits],
                     "generated": generated, "generated_text": tokenizer.decode(generated)})

reference = {
    "model": args.model, "folder": os.path.abspath(folder), "transformers_device": device,
    "adapter": os.path.abspath(args.adapter) if args.adapter else None,
    "texts": [{"text": t, "ids": (i := tokenizer(t, add_special_tokens=False).input_ids), "decoded": tokenizer.decode(i)} for t in texts],
    "tools": tools, "chats": chats, "runs": runs,
}
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(reference, f, ensure_ascii=False)
print(f"wrote {args.out}: {len(texts)} texts, {len(chats)} chats, {len(runs)} prompts with logits ({args.model} in {folder})")
