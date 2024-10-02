# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import Counter, defaultdict
from typing import Tuple
import math
from data.charloader import load_chars_from_file

# PYTHON PROJECT IMPORTS
from utils import START_TOKEN, END_TOKEN, Vocab, read_mono
from ngram import Ngram

class CharPredictor():
    def __init__(self, n, charmap_path, train_path):
        self.n = n

        train_data = load_chars_from_file(train_path)
        self.model = Ngram(n, train_data)

        charmap_data = load_chars_from_file(charmap_path)

        # correct types from starter code
        self.english_to_mandarin = defaultdict(set)
        self.mandarin_to_english = dict()

        for line in charmap_data:
            eng = ''.join(line[2:])
            man = line[0]

            self.english_to_mandarin[eng].add(man)
            self.mandarin_to_english[man] = eng

        # self.oenglish_to_mandarin = defaultdict(set)
        # self.omandarin_to_english = dict()
        # charmap_lines = read_mono(charmap_path, delim=" ")

        # i = 0
        # for _, c_char, e_pron, _ in charmap_lines:
        #     if (i % 8000 == 0):
        #         print(f"{c_char} and {e_pron}")
        #     i += 1
        #     self.oenglish_to_mandarin[e_pron].add(c_char)
        #     self.omandarin_to_english[c_char] = e_pron

        # assert self.oenglish_to_mandarin == self.english_to_mandarin
        # assert self.omandarin_to_english == self.mandarin_to_english

    def candidates(self, token: str) -> Sequence[str]:
        pronunciations = self.english_to_mandarin[token]
        
        if len(token) == 1:
            pronunciations.add(token)
        elif token == "<space>":
            pronunciations.add(" ")
        
        return pronunciations

    def start(self) -> Sequence[str]:
        return self.model.start()
    
    def step(self, q: Sequence[str], w: str) -> Tuple[Sequence[str], Mapping[str, float]]:
        # state transition will be the same
        r = q

        p = self.model.logprobs.get(r, {v: math.log(1/(len(self.model.vocab)-1)) for v in self.model.vocab if v != START_TOKEN})

        candidates_p = dict()
        for c in self.candidates(w):
            candidates_p[c] = p.get(c, -math.inf)

        return r, candidates_p

if __name__ == "__main__":
    my_model = CharPredictor(1, "data/mandarin/charmap", "data/mandarin/train.han")
