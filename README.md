# Distilling Hypernymy Relations from Language Models

This is the official repository of the \*SEM 2022 paper **_Distilling Hypernymy Relations from Language Models: On the Effectiveness of Zero-Shot Taxonomy Induction_**. We investigate the use of pretrained language models (LM) for taxonomy learning in a zero-shot setting using prompting and sentence-scoring methods. Through extensive experiments on public benchmarks from [TExEval-1](https://www.aclweb.org/anthology/S15-2151/) and [TExEval-2](https://www.aclweb.org/anthology/S16-1168/), we show that our proposed approaches outperform some supervised methods and are competitive with SOTA under certain conditions.

Paper is available on [arXiv](https://arxiv.org/abs/2202.04876).

## Setup

### Create `conda` environment

`conda create -n taxonomy -y python=3.7 && conda activate taxonomy`

### Install Dependencies

1. Install the required packages
   `pip install -r requirements.txt`

2. Install [MXNet](https://mxnet.apache.org/versions/1.8.0/get_started?platform=linux&language=python&processor=gpu&environ=pip&) based on CUDA version

```
nvcc --version        # to check CUDA version
pip install <mxnet>   # corresponding MXNet version
```

## Run Experiments

The experiments can be run via a bash script that generates and evaluates taxonomies using a single command.

```
./run.sh <method_name> <model_checkpoint> <domain> <prompt_type>
```

Here,

- method_name: `{prompt-mlm, restrict-mlm, lm-scorer}`
- model_checkpoint: `{bert-base-uncased, bert-large-uncased, roberta-base, roberta-large}` (`gpt2` and `gpt2-medium` can also be used for LMScorer)
- domain: `{equipment, environment, food, science_ev, science_wn, science}`
- prompt_type: `{gen, spec, type}`

The taxonomies are generated in the directory `output/{method_name}/{model_checkpoint}` and the corresponding results are saved as `results/{method_name}.csv`.

Currently, taxonomies with top-k hypernyms for each term are generated where k in {1, 3, 5}.

## Citation

If our research helps you, please kindly cite our paper:

```
@article{Jain2022DistillingHR,
  title={Distilling Hypernymy Relations from Language Models: On the Effectiveness of Zero-Shot Taxonomy Induction},
  author={Devansh Jain and Luis Espinosa Anke},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.04876}
}
```

## Acknowledgement

The code is implemented using [transformers](https://github.com/huggingface/transformers) and [mlm-scoring](https://github.com/awslabs/mlm-scoring).
