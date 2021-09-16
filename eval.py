"""
Evaluation script for computing precision, recall and F1 scores.
"""

import argparse
import logging
import networkx as nx
import pandas as pd

from pathlib import Path


def post_process(df):
    """
    Post-processing for taxonomies.
    Currently: lowercase all entitites
    """
    cols = ["Hyponym", "Hypernym"]
    for col in cols:
        df[col] = df[col].str.lower()
    return df


def build_graph(taxo):
    """
    Build a networkx graph of the taxonomy for edge computations.
    """
    df = pd.read_csv(taxo, sep='\t', names=["Hyponym", "Hypernym"])
    df = post_process(df)
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_edge(row["Hyponym"], row["Hypernym"])
    return G


def compute_edge_metrics(predicted_taxo, gold_taxo):
    """
    Args:
        predicted_taxo(str): model-generated taxonomy
        gold_taxo (str): corresponding gold-standard taxonomy
    Returns:
        recall (float), precision(float), f1 (float): edge metrics for the taxonomies
    """
    predicted_graph = build_graph(predicted_taxo)
    predicted_edges = predicted_graph.edges
    gold_graph = build_graph(gold_taxo)
    gold_edges = gold_graph.edges

    # tp -> true positive, fp -> false positive, fn -> false negative
    tp = len([edge for edge in predicted_edges if edge in gold_edges])
    fp = len([edge for edge in predicted_edges if edge not in gold_edges])
    fn = len([edge for edge in gold_edges if edge not in predicted_edges])

    if tp == 0:
        return 0, 0, 0

    recall = round(tp / (tp + fn), 3) * 100
    precision = round(tp / (tp + fp), 3) * 100
    f1 = 2 * (recall * precision) / (recall + precision)
    f1 = round(f1, 3)
    return recall, precision, f1


if __name__ == "__main__":
    # logging
    transformer_logging = logging.getLogger('transformers')
    transformer_logging.setLevel(logging.ERROR)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--model-name", default='bert-base-uncased', type=str)
    parser.add_argument("--domain", default="science_ev", type=str)
    parser.add_argument("--prompt", default="type", type=str)

    args = parser.parse_args()
    logging.info(f"Args: {args}")

    top_n = [1, 3, 5]
    prompt = args.prompt

    stats = {
        'Domain': [],
        'Model': [],
        'Prompt': [],
        'Top-N': [],
        'Precision': [],
        'Recall': [],
        'F1': []
    }

    logging.info(f"Evaluating {args.method_name} for {args.domain}...")
    gold_taxo = f"{Path.cwd()}/data/gold/{args.domain}.taxo"
    for n in top_n:
        logging.info(f"{args.domain}-{prompt}-{n}.taxo")
        predicted_taxo = f"{Path.cwd()}/output/{args.method_name}/{args.model_name}/{args.domain}-{prompt}-{n}.taxo"
        r, p, f1 = compute_edge_metrics(predicted_taxo, gold_taxo)
        stats['Domain'].append(args.domain)
        stats['Model'].append(args.model_name)
        stats['Prompt'].append(prompt)
        stats['Top-N'].append(n)
        stats['Precision'].append(p)
        stats['Recall'].append(r)
        stats['F1'].append(f1)
    logging.info(f"Evaluation finished for {args.domain}!")

    stats_df = pd.DataFrame(stats)
    # if a results file already exists, append new values
    # else create a results file
    stats_path = Path(f"{Path.cwd()}/results/{args.method_name}.csv")
    if stats_path.is_file():
        temp_df = pd.read_csv(stats_path)
        stats_df = temp_df.append(stats_df)
    stats_df.to_csv(stats_path, index=False)
    logging.info(f"Results saved for {args.domain} to {stats_path}!")
