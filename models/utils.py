from pathlib import Path


def create_dir(method_name, model_name):
    dir_method = Path(f"{Path.cwd()}/output/{method_name}/{model_name}")
    if not dir_method.exists():
        dir_method.mkdir(parents=True, exist_ok=True)


def get_terms(domain):
    terms = []
    path = f"{Path.cwd()}/data/terms/{domain}.terms"
    with open(path, 'r') as f:
        for line in f:
            # extract term
            term = line.split('\t')[1]
            # remove '\n'
            term = term[:-1]
            terms.append(term)
    return terms