# Taken from https://stackoverflow.com/a/36898903/3776827
# Iterate in the current directory to list sub-dir and files
import os

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
         if filename.endswith(".asm") or filename.endswith(".py"):
             # print(os.path.join(directory, filename))
            r.append(os.path.join(root, filename))
    return r
