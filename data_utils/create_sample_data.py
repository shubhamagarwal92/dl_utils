import pandas as pd
import torch
from transformers import AutoTokenizer


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



def create_causal_mask(seq_len: int) -> torch.BoolTensor:
    """
    Create a causal mask for sequence of length seq_len.
    Shape: (seq_len, seq_len)
    True means allowed attention, False means masked.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.BoolTensor:
    """
    Create padding mask for input_ids.
    Shape: (batch_size, seq_len)
    True for valid tokens, False for padding tokens.
    """
    return input_ids != pad_token_id


def create_combined_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.BoolTensor:
    """
    Create combined causal + padding mask.
    Shape: (batch_size, seq_len, seq_len)
    True means allowed attention, False means masked.
    """
    batch_size, seq_len = input_ids.size()
    
    causal_mask = create_causal_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1)    # (batch, seq_len, seq_len)
    padding_mask = create_padding_mask(input_ids, pad_token_id)                         # (batch, seq_len)
    padding_mask_keys = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)               # (batch, seq_len, seq_len)
    
    combined_mask = causal_mask & padding_mask_keys
    return combined_mask


def tokenize_and_create_mask(sentences, tokenizer_name="bert-base-uncased"):
    """
    Tokenizes input sentences with padding and returns input_ids and combined attention mask.
    
    Args:
        sentences (list of str): List of text sequences.
        tokenizer_name (str): HuggingFace tokenizer model name.
    
    Returns:
        input_ids (torch.Tensor): token IDs padded.
        attention_mask (torch.BoolTensor): combined causal + padding mask.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    pad_token_id = tokenizer.pad_token_id
    
    combined_mask = create_combined_attention_mask(input_ids, pad_token_id)
    return input_ids, combined_mask

def example_wrapper():
    # ------------- Example usage -------------
    
    sentences = ["hey", "the cat sat", "hello world"]
    input_ids, attention_mask = tokenize_and_create_mask(sentences)
    
    print("Input IDs:")
    print(input_ids)
    
    print("\nCombined Causal + Padding Mask:")
    print(attention_mask)


