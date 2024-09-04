import os
import time
from functools import lru_cache

from fastapi import FastAPI

if "TRANSFORMERS_CACHE" not in os.environ:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from constants import TRANSFORMERS_CACHE

    os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser(
        os.sep.join(TRANSFORMERS_CACHE.split("/"))  # before importing transformers
    )

import torch
import torch.nn.functional as F
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)


@lru_cache(maxsize=None)
def get_model_and_tokenizer():

    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = [
        "gpt-j-6B",
        "opt-66b",
        "gpt-neox-20b",
        "T0pp",
        "opt-125m",
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",
        "flan-t5-base-bf16",
        "flan-t5-large-bf16",
        "flan-t5-xl-bf16",
        "flan-t5-xxl-bf16",
    ]
    assert model_shortname in valid_model_shortnames, f"Model name {model_shortname} not in {valid_model_shortnames}"

    if model_shortname == "gpt-j-6B":

        model_name = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-66b":

        model_name = "facebook/opt-66b"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname == "gpt-neox-20b":

        model_name = "EleutherAI/gpt-neox-20b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="main",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "T0pp":

        model_name = "bigscience/T0pp"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-125m":

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname.startswith("flan-t5") and "bf16" not in model_shortname:
        model_name = "google/" + model_shortname
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", device_map=hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname.startswith("flan-t5") and model_shortname.endswith("-bf16"):

        assert torch.cuda.is_bf16_supported()
        assert is_torch_bf16_gpu_available()
        model_name = "google/" + model_shortname.replace("-bf16", "")
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map=hf_device_map, torch_dtype=torch.bfloat16
        )
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    elif model_shortname == "ul2":
        model_name = "google/" + model_shortname
        model = T5ForConditionalGeneration.from_pretrained(
            # Don't use auto here. It's slower cpu loading I guess.
            "google/ul2"  # , low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")

    return model, tokenizer


class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert (
            len(self.tokenizer.encode(eos_text)) < 10
        ), "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        return condition1 or condition2


app = FastAPI()


@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {"message": f"Hello! This is a server for {model_shortname}. " "Go to /generate/ for generation requests."}


def apply_top_p_sampling(logits, top_p=0.9):
    """Applies top-p (nucleus) sampling to filter logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')  # Set removed tokens to a very low value
    return logits

def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """Applies repetition penalty to discourage repeated tokens."""
    for i, token_set in enumerate(generated_tokens):
        for token in token_set:
            if logits[i, token] < 0:
                logits[i, token] *= penalty
            else:
                logits[i, token] /= penalty
    return logits

@app.get("/generate/")
async def generate(
    prompt: str,
    prompt_without_context: str = "",
    max_input: int = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    eos_text: str = None,
    keep_prompt: bool = False,
    alpha: float = 0.3,  # Context weight for CAD
):
    start_time = time.time()
    model_shortname = os.environ["MODEL_NAME"]

    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_input).cuda()
    inputs_without_context = tokenizer.encode(prompt_without_context, return_tensors="pt", max_length=max_input).cuda()

    stopping_criteria_list = StoppingCriteriaList()
    if eos_text:
        stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=eos_text)
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

    is_encoder_decoder = model_shortname in ["T0pp", "ul2"] or model_shortname.startswith("flan-t5")

    # Initialize generated sequences and logits storage
    generated_ids = inputs
    generated_ids_without_context = inputs_without_context

    for _ in range(max_length):
        # Generate logits for the next token with context
        outputs_with_context = model(
            input_ids=generated_ids, return_dict=True, output_logits=True
        )
        logits_with_context = outputs_with_context.logits[:, -1, :]  # Take the logits of the last token

        # Generate logits for the next token without context
        outputs_without_context = model(
            input_ids=generated_ids_without_context, return_dict=True, output_logits=True
        )
        logits_without_context = outputs_without_context.logits[:, -1, :]  # Take the logits of the last token

        # Apply the CAD formula
        combined_logits = ((1 + alpha) * logits_with_context) - (alpha * logits_without_context)

        # Apply softmax to convert logits into probabilities
        probs = F.softmax(combined_logits, dim=-1)

        # Sample or select the next token
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # Append the generated token to the sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        generated_ids_without_context = torch.cat([generated_ids_without_context, next_token], dim=1)

        # Stop if end-of-sequence is reached
        if eos_text and tokenizer.decode(next_token.item()).strip() == eos_text:
            break

    # Decode the final sequences
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    generated_num_tokens = [len(generated_ids_) for generated_ids_ in generated_ids]
    if not keep_prompt and not is_encoder_decoder:
        generated_texts = [
            generated_text[generated_text.index(prompt) + len(prompt):] for generated_text in generated_texts
        ]
    elif keep_prompt and is_encoder_decoder:
        generated_texts = [prompt + generated_text for generated_text in generated_texts]

    end_time = time.time()
    run_time_in_seconds = end_time - start_time
    return {
        "generated_num_tokens": generated_num_tokens,
        "generated_texts": generated_texts,
        "run_time_in_seconds": run_time_in_seconds,
        "model_name": model_shortname,
    }

print("\nLoading model and tokenizer.")
get_model_and_tokenizer()
print("Loaded model and tokenizer.\n")
