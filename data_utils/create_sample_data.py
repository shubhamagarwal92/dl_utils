import pandas as pd
import torch


def get_pd_df():
  """
  https://pandas.pydata.org/docs/user_guide/10min.html
  """
  df2 = pd.DataFrame(
      {
          "A": 1.0,
          "B": pd.Timestamp("20130102"),
          "C": pd.Series(1, index=list(range(4)), dtype="float32"),
          "D": np.array([3] * 4, dtype="int32"),
          "E": pd.Categorical(["test", "train", "test", "train"]),
          "F": "foo",
      }
  )
  
  print(df2.dtypes)


def get_torch_tensor():
  logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 1.5])  # shape: (num_candidates,)
  return logits


def test_mask():
  # x = torch.randn(1, 5)
  x = torch.tensor([2,3,4,5,0,0])
  mask = torch.eq(x, 0)
  
  print(x)
  print(mask)
  scores = torch.tensor([2.4,3.5,4,5,6,7])
  
  masked_x = scores.masked_fill(mask, 0)
  print(masked_x)
  
  # x.masked_fill_(mask, float('-inf'))
  
