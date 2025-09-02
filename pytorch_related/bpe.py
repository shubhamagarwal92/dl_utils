# Simple BPE - Credits GPT
from collections import defaultdict
from typing import List, Tuple

class BPETokenizer:
    def __init__(self, num_merges: int = 10):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab = {}

    def get_vocab(self, corpus: List[str]):
        """Initial vocab: words split into chars with </w> marker."""
        vocab = {}
        for word in corpus:
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] = vocab.get(word, 0) + 1
        return vocab

    def get_stats(self, vocab):
        """Count frequency of adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Merge all occurrences of a given pair in the vocab."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus: List[str]):
        """Learn BPE merges from the corpus."""
        self.vocab = self.get_vocab(corpus)
        for i in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best_pair, self.vocab)
            self.merges.append(best_pair)
            print(f"Step {i+1}: Merged {best_pair}")

    def encode_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word."""
        symbols = list(word) + ["</w>"]
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            merge_candidates = {pair: i for i, pair in enumerate(pairs) if pair in self.merges}
            if not merge_candidates:
                break
            # find earliest merge in order of learned rules
            merge_index = min(self.merges.index(pair) for pair in merge_candidates)
            merge_pair = self.merges[merge_index]
            # perform merge
            new_symbols = []
            skip = False
            for i in range(len(symbols)):
                if skip:
                    skip = False
                    continue
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == merge_pair:
                    new_symbols.append(''.join(merge_pair))
                    skip = True
                else:
                    new_symbols.append(symbols[i])
            symbols = new_symbols
        # remove </w> for readability
        if symbols[-1] == "</w>":
            symbols = symbols[:-1]
        return symbols

    def encode(self, text: str) -> List[str]:
        """Encode a space-separated string."""
        words = text.split()
        return [self.encode_word(word) for word in words]

    def decode(self, tokens: List[List[str]]) -> str:
        """Reconstruct text from encoded tokens."""
        words = ["".join(word_tokens) for word_tokens in tokens]
        return " ".join(words)


    def get_final_vocab(self) -> List[str]:
        """Return the set of learned subwords as final vocab."""
        subwords = set()
        for word in self.vocab:
            subwords.update(word.split())
        return sorted(subwords)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    corpus = ["low", "lower", "newest", "widest"]
    tokenizer = BPETokenizer(num_merges=10)
    tokenizer.fit(corpus)

    print("\nLearned merges:", tokenizer.merges)

    # Encoding new words
    print("\nEncoding examples:")
    print("lowest  ->", tokenizer.encode("lowest"))
    print("newer   ->", tokenizer.encode("newer"))
    print("widest  ->", tokenizer.encode("widest"))

    # Encoding and decoding
    text = "lowest newer widest"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("\nEncoding:", encoded)
    print("Decoded :", decoded)

    # Final vocab
    print("\nFinal Vocab:", tokenizer.get_final_vocab())

