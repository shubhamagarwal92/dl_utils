# Taken from
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/openai_transformer.py

import torch.nn as nn
import torch
import math
from typing import NamedTuple, List


class Swish(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)



from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"



class RandomDataset(Dataset):

    def __init__(self):
        ...
        ...

        self.elmo = Elmo(config["options_file"], config["weight_file"],
                 2, dropout=config["elmo_dropout"])


    def __getitem__(self, index):
        ...
        ...

        ques_embeddings = []
        for i in range(len(dialog)):
            ques_char_ids = batch_to_ids(dialog[i]["question"])
            ques_embeddings.append(self.elmo(ques_char_ids)['elmo_representations'][0])

        ques_embeddings = torch.stack(ques_embeddings,0)
