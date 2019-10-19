from src.pipeline_utils.validation import stratified_split
import pandas as pd
from pathlib import Path
import os

DATA_PATH = Path("../data")

df = pd.read_csv(DATA_PATH / "reddit_train.csv")
df_train, df_valid = stratified_split(df, "subreddits")

lines_train = [f"__label__{l} {c}"for c,l in zip(df_train.comments, df_train.subreddits)]
lines_train = [l.replace("\n", " ").replace("\t", " ") for l in lines_train]
lines_valid = [f"__label__{l} {c}"for c,l in zip(df_valid.comments, df_valid.subreddits)]
lines_valid = [l.replace("\n", " ").replace("\t", " ") for l in lines_valid]

with open(DATA_PATH / "fasttext.train.txt", "w") as f:
    f.write("\n".join(lines_train))

with open(DATA_PATH / "fasttext.valid.txt", "w") as f:
    f.write("\n".join(lines_valid))

os.chdir(DATA_PATH)
os.system("cat fasttext.train.txt | sed -e \"s/\([.\!?,’/()]\)/ \\1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > fasttext.train.processed.txt")
os.system("cat fasttext.valid.txt | sed -e \"s/\([.\!?,’/()]\)/ \\1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > fasttext.valid.processed.txt")