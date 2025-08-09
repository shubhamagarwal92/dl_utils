# Copied from https://github.com/LitLLM/litllms-for-literature-review-tmlr/blob/main/generation/autoreview/models/data_utils.py
from functools import partial
from transformers import AutoTokenizer
import datasets
from datasets import set_caching_enabled, Dataset, load_dataset
import pandas as pd


def compute_length(example, col_name):
    # https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt#creating-new-columns
    example[f"{col_name}_length"] = len(example[col_name].split())
    return example


def length_stats(dataset, col_name: str = "related_work"):
    dataset = dataset.map(partial(compute_length, col_name=col_name))
    # https://discuss.huggingface.co/t/copy-columns-in-a-dataset-and-compute-statistics-for-a-column/22157/11
    mean = dataset.with_format("pandas")[f"{col_name}_length"].mean()
    print(mean)

def get_hf_dataset(dataset_name, small_dataset: bool = False, split: str= "test", redownload: bool = False):
    if redownload:
        dataset = load_dataset(dataset_name, split=split, download_mode='force_redownload')
    else:
        dataset = load_dataset(dataset_name, split=split)
    hf_column_names = dataset.column_names
    if "ref_abstract_full_text" in hf_column_names:
        dataset = dataset.remove_columns(['ref_abstract_full_text_original', 'ref_abstract_full_text', "ref_abstract_original"])
    if small_dataset:
        dataset = dataset.select(list(range(5)))
    set_caching_enabled(False)
    return dataset
