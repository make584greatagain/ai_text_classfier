import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def load_and_prepare_data(file_path: str, downsample_n: int = 5000) -> DatasetDict:
    df = pd.read_csv(file_path)
    df["sentence"] = df["sentence"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = df["label"].astype(int)

    df = df.groupby("label").apply(lambda x: x.sample(n=min(downsample_n, len(x)))).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    train_dataset = Dataset.from_pandas(train_df[["sentence", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["sentence", "label"]])

    return DatasetDict({"train": train_dataset, "test": test_dataset})
