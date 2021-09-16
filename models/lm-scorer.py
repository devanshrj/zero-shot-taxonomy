"""
Language Model (LM) sentence scoring using prompts.
"""

import argparse
import logging
import mxnet as mx
import torch
import warnings

from mlm.scorers import MLMScorer, LMScorer
from mlm.models import get_pretrained
from pathlib import Path
from tqdm import tqdm

from utils import *


PROMPTS = {
    'gen': "<mask> is more general than <term>.",
    'spec': "<term> is more specific than <mask>.",
    'type': "<term> is a type of <mask>."
}


def get_prompt(prompt, hypo, hyper):
    prompt = PROMPTS[prompt]
    prompt = prompt.replace('<term>', hypo)
    prompt = prompt.replace('<mask>', hyper)
    return prompt


def load_pretrained(model_name, device):
    """
    Load pretrained HuggingFace tokenizer and model.
    :param model_name: model checkpoint
    :param device: CPU / CUDA device
    :return: tokenizer and model
    """
    logging.info(f"Initialising {model_name}")
    model, vocab, tokenizer = get_pretrained(device, model_name)
    if 'gpt2' in model_name:
        scorer = LMScorer(model, vocab, tokenizer, device)
    else:
        scorer = MLMScorer(model, vocab, tokenizer, device)
    return scorer


def get_sentences(prompt, hypo, terms):
    sentences = []
    options = []
    for hyper in terms:
        if hypo == hyper:
            continue
        text = get_prompt(prompt, hypo, hyper)
        sentences.append(text)
        options.append(hyper)
    return sentences, options


def get_taxo(args, scorer, domain, prompt):
    dir = f"{Path.cwd()}/output/{args.method_name}/{args.model_name}"
    f1 = open(f'{dir}/{domain}-{prompt}-1.taxo', 'w')
    f3 = open(f'{dir}/{domain}-{prompt}-3.taxo', 'w')
    f5 = open(f'{dir}/{domain}-{prompt}-5.taxo', 'w')

    terms = get_terms(domain)

    logging.info(f"Creating taxonomy for {domain}-{prompt}...")
    total = 0
    progress_bar = tqdm(terms, desc=f'{domain} + {prompt}')
    for focus in progress_bar:
        sentences, options = get_sentences(prompt, focus, terms)

        with warnings.catch_warnings():
            # hacky way to disable logging info and warnings for FixedBucketSampler from mlm
            warnings.simplefilter("ignore")
            logging.disable(logging.INFO)
            scores = scorer.score_sentences(sentences)
            logging.disable(logging.NOTSET)
        scores = torch.Tensor(scores)

        top_scores = scores.topk(10)
        indices = top_scores[1].cpu().numpy()

        count = 0
        for i in indices:
            word = options[i]
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
    parser.add_argument("--model-name", default="gpt2-117m-en-cased", type=str)
    parser.add_argument("--method-name", default="lm-scorer", type=str)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--prompt", default="type", type=str)

    args = parser.parse_args()
    logging.info(f"Args: {args}")

    # utils
    create_dir(args.method_name, args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'cuda' in device.type:
        ctxs = [mx.gpu(0)]
    else:
        ctxs = [mx.cpu()]
    logging.info(f"Device: {ctxs}")

    # map the model names (input: transformers, output: mxnet) to provide a uniform interface
    model_map = {
        'bert-base-uncased': 'bert-base-en-uncased', 'bert-large-uncased': 'bert-large-en-uncased',
        'roberta-base': 'roberta-base-en-cased', 'roberta-large': 'roberta-large-en-cased',
        'gpt2': 'gpt2-117m-en-cased', 'gpt2-medium': 'gpt2-345m-en-cased'
    }
    model_name = model_map[args.model_name]

    # tokenizer and model initialisation
    scorer = load_pretrained(model_name, ctxs)

    logging.info(f"==================================={args.domain}===================================")
    get_taxo(args, scorer, args.domain, args.prompt)