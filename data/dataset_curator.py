import os, random
from utils.io import dir_files, copy_files, create_folder, zip_dir
"""
In https://github.com/tushartushar/DeepLearningSmells, the provided training data was not all used for model training.
This script files uses the subsets' sizes that were mentioned in 10.1016/j.jss.2021.110936 (P9, Table 2). It does so by taking as input 
the files located in the aforementioned repository, and randomly pick the apropriate number of files mentioned in the study.
Note that files that were not picked will be deleted.

`data_path` should honor the following structure:
    - data_path:
        -- CM
            --- Postitive
            --- Negative
        -- CC
            --- Positive
            --- Negative
        -- FE
            --- Positive
            --- Negative
        -- MFA
            --- Positive
            --- Negative
"""
data_path = ""
configs = [
    {"smell": "CM", "Positive": 12489, "Negative": 139960},
    {"smell": "CC", "Positive": 6186, "Negative": 149767},
    {"smell": "FE", "Positive": 1788, "Negative": 51260},
    {"smell": "MFA", "Positive": 290, "Negative": 50205},
]

def curator():
    # For each smell
    for smell in configs:
        smell, pos, neg = smell.values()
        # Randomly pick pos and neg
        pos_dir = set(dir_files(os.path.join(data_path, smell, "Positive")))
        neg_dir = set(dir_files(os.path.join(data_path, smell, "Negative")))
        sampled_pos = random.sample(pos_dir, pos)
        sampled_neg = random.sample(neg_dir, neg)

        new_dir = f"./experiment_{smell}"
        new_pos_dir = os.path.join(new_dir, "Positive")
        new_neg_dir = os.path.join(new_dir, "Negative")
        
        create_folder(new_pos_dir)
        create_folder(new_neg_dir)

        copy_files(sampled_pos, new_pos_dir)
        copy_files(sampled_neg, new_neg_dir)

        zip_dir(new_pos_dir, f'{new_dir}.7z')


        

if __name__ == "__main__":
    curator()