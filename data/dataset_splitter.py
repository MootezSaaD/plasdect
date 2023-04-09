import pandas as pd
import os, random
from utils.io import load_files_by_ext

_path = "/home/mootez/scratch/code_smells_dataset"

smells_configs = [
{
 "folder": "experiment_ComplexConditional",
 "training": {"Positive": 5000, "Negative": 5000},
 "eval": {"Positive": 7489 , "Negative": 139460}
 
},
{
 "folder": "experiment_ComplexMethod",
 "training": {"Positive": 4330, "Negative": 4330},
 "eval": {"Positive": 1856, "Negative": 145437}
 
},
{
 "folder": "experiment_FeatureEnvy",
 "training": {"Positive": 1260, "Negative": 1260},
 "eval": {"Positive": 528, "Negative":  50000}
 
},
{
 "folder": "experiment_MultifacetedAbstraction",
 "training": {"Positive": 205, "Negative": 205},
 "eval": {"Positive": 85, "Negative":  50000}
 
}
]

def _dataset_splitter(config, root_path):
    """
    Returns pandas dataframe with column: "file", "label", "split"
    """
    # Load positive samples
    pos = load_files_by_ext(os.path.join(root_path, config["folder"], "Positive"), "symb")
    # Load negative samples
    neg = load_files_by_ext(os.path.join(root_path, config["folder"], "Negative"), "symb")
    # Shuffle each
    random.shuffle(pos)
    random.shuffle(neg)

    train_pos = random.sample(pos, config["training"]["Positive"])
    train_neg = random.sample(neg, config["training"]["Negative"])

    eval_pos = list(set(pos) - set(train_pos))
    eval_neg = list(set(neg) - set(train_neg))

    df = [(f, 1) for f in train_pos] + [(f, 0) for f in train_neg] + [(f, 1) for f in eval_pos] + [(f, 0) for f in eval_neg]

    df = pd.DataFrame(df)
    df.columns = ["file", "label"]

    return df


if __name__ == "__main__":
    for s in smells_configs:
        df = _dataset_splitter(s, _path)
        df.to_csv(os.path.join(_path, f"{s['folder']}.csv"), index=None)
