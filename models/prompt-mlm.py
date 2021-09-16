"""
MaskedLM with prompts.
"""

import argparse
import logging
import torch

from pathlib import Path
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

from utils import *


PROMPTS = {
    'gen': "<mask> is more general than <term>.",
    'spec': "<term> is more specific than <mask>.",
    'type': "<term> is a type of <mask>."
}


def get_prompt(prompt, focus, mask):
    prompt = PROMPTS[prompt]
    prompt = prompt.replace('<term>', focus)
    prompt = prompt.replace('<mask>', mask)
    return prompt


def load_pretrained(model_name, device):
    """
    Load pretrained HuggingFace tokenizer and model.
    :param model_name: model checkpoint
    :param device: CPU / CUDA device
    :return: tokenizer and model
    """
    logging.info(f"Initialising {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_taxo(args, model, tokenizer, device, domain, prompt):
    dir = f"{Path.cwd()}/output/{args.method_name}/{args.model_name}"
    f1 = open(f'{dir}/{domain}-{prompt}-1.taxo', 'w')
    f3 = open(f'{dir}/{domain}-{prompt}-3.taxo', 'w')
    f5 = open(f'{dir}/{domain}-{prompt}-5.taxo', 'w')

    terms = get_terms(domain)

    logging.info(f"Creating taxonomy for {domain}-{prompt}...")
    total = 0
    progress_bar = tqdm(terms, desc=f'{domain} + {prompt}')
    for focus in progress_bar:
        text = get_prompt(prompt, focus, tokenizer.mask_token)
        inputs = tokenizer(text, return_tensors = "pt").to(device)
        mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        
        output = model(**inputs)
        word_preds = output.logits
        mask_preds = word_preds[0, mask_index]

        top_ids = mask_preds.topk(10)[1]
        top_tokens = tokenizer.convert_ids_to_tokens(top_ids.cpu().numpy())

        count = 0
        for token in top_tokens:            
            word = token.lower().strip()
            # for SentencePiece tokenizers
            word = word.replace('ġ', '')
            if not word.isalpha() or word == focus.lower():
                continue
            
            out = f"{total}\t{focus}\t{word}\n"
            
            count += 1
            total += 1
            # required number of terms found
            if count <= 1:
                f1.write(out)
                f3.write(out)
                f5.write(out)
            elif 1 < count <= 3:
                f3.write(out)
                f5.write(out)
            elif 3 < count <= 5:
                f5.write(out)
            else:
                break
    logging.info("Taxonomy created!")


if __name__ == "__main__":
    transformer_logging = logging.getLogger('transformers')
    transformer_logging.setLevel(logging.ERROR)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="bert-base-uncased", type=str)
    parser.add_argument("--method-name", default="prompt-mlm", type=str)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--prompt", default="type", type=str)

    args = parser.parse_args()
    logging.info(f"Args: {args}")

    # utils
    create_dir(args.method_name, args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # tokenizer and model initialisation
    tokenizer, model = load_pretrained(args.model_name, device)

    logging.info(f"==================================={args.domain}===================================")
    get_taxo(args, model, tokenizer, device, args.domain, args.prompt)