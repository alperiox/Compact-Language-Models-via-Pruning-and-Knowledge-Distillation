class Tokenizer:
    def __init__(self, chars: list[str] | str):
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ixs: list[int]) -> str:
        return "".join([self.itos[ix] for ix in ixs])
