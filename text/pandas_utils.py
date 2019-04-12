import re
import numpy as np
from pandas.io.json import json_normalize
import json
import pandas as pd
import time
import os
import glob
import argparse
import string
import unicodedata


def read_json_to_df(file_path):
	# df = pd.read_json(path_or_buf=file_path,orient='records',lines=True)
	df = pd.read_json(path_or_buf=file_path,orient='records')
	return df

def flatten_json_column(df,col_name='utterance'):
	temp_df = json_normalize(df[col_name].tolist())
	df.reset_index(drop=True,inplace=True)
	df = df.join(temp_df).drop(col_name, axis=1)
	return df

def get_column_stats(df,column_name,to_dict = False):
	if to_dict:
		return df[column_name].value_counts().to_dict()
	else:
		return df[column_name].value_counts()

def findFiles(path):
	return glob.glob(path)

def get_column_names(df):
	return df.columns.values

def get_value_row_column(df,index,column_name):
	return df.get_value(index,column_name)

def flatten_dic_column(df,col_name):
	df_new= pd.concat([df.drop([col_name], axis=1), df[col_name].apply(pd.Series)], axis=1)
	return df_new

def append_df(df, df_to_append, ignore_index=True):
	new_df = df.append(df_to_append,ignore_index=ignore_index)
	return new_df

def write_df_to_csv(df,outputFilePath):
	df.to_csv(outputFilePath, sep=str('\t'),quotechar=str('"'), index=False, header=True)

def write_df_to_json(df,outputFilePath):
	df.to_json(path_or_buf=outputFilePath,orient='records',lines=True)

def save_df_pickle(df,output_file):
	df.to_pickle(output_file)

def get_unique_column_values(df,col_name):
	""" Returns unique values """
	return df[col_name].unique()

def count_unique(df, col_name):
	""" Count unique values in a df column """
	count = df[col_name].nunique()
	return count

def main(args):
    start = time.time()
    stats_df = pd.DataFrame()
    global_df = pd.DataFrame()
    print("Reading files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_dir', help='Input file directory path')
    args = parser.parse_args()
    main(args)
