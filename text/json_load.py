import pandas as pd
# from pathlib import Path
from datetime import date
import calendar
import json
import warnings

warnings.filterwarnings('ignore')
import glob
import pprint
import numpy as np
import argparse
import os
import pdb

class DataLoader(object):
    def __init__(self, data_path):
        super(DataLoader, self).__init__()
        self.all_data = pd.DataFrame()
        # Python3
        # data_path = Path(data_path)
        # for filename in data_path.glob("*.json"):
        for filename in glob.glob(data_path + "/*.json"):
            print(filename)
            with open(filename, 'r') as file:
                data = json.load(file)
                # If we get ValueError/DecodeError: Extra data
                # JSONDecodeError: Extra data: line 2 column 1
                # https://stackoverflow.com/a/29312618/3776827
                # data = []
                # for line in file:
                #     data.append(json.loads(line))
            temp_df = pd.DataFrame(data)
            self.all_data = self.all_data.append(temp_df, ignore_index=True)

    @property
    def get_size(self):
        print("Total size")
        print(len(self.all_data))

def main():
    parser = argparse.ArgumentParser(description='Utility Functions')
    parser.add_argument('-data_path', type=str, default="",
                        help="path to input")
    args = parser.parse_args()
    loader = DataLoader(args.data_path)
    loader.get_size(args.data_path)


if __name__ == "__main__":
    main()
