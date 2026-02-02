"""
utils.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def setup_logging(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def load_dataset(csv_path, val_split=0.1, shuffle=True):
    df = pd.read_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if val_split > 0:
        train_df, val_df = train_test_split(df, test_size=val_split)
        return train_df, val_df
    else:
        return df, None