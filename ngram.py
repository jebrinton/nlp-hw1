# # SYSTEM IMPORTS
# from collections.abc import Mapping, Sequence
# from collections import Counter, defaultdict
# from typing import Tuple
# import math

# # PYTHON PROJECT IMPORTS
# from utils import START_TOKEN, END_TOKEN, Vocab


# class Uniform(object):
#     def __init__(self, vocab):
#         self.vocab = vocab

#         self.uniform_dict = {v: math.log(1/(len(self.vocab)-1)) for v in self.vocab if v != utils.START_TOKEN}
#         self.logprobs = defaultdict(lambda: self.uniform_dict)

# class Ngram(object):
#     """An ngram language model.

#     data: a list of lists of symbols. They should not contain `<EOS>`;
#           the `<EOS>` symbol is automatically appended during
#           training.
#     """
    
#     def __init__(self,
#                 n: int,
#                 data: Sequence[Sequence[str]]
#                 ) -> None:
#         self.n = n
#         self.vocab = Vocab()
#         count: collections.Counter = collections.Counter()
#         total: int = 0
#         for line in data:
#             for a in list(line) + [END_TOKEN]:
#                 self.vocab.add(a)
#                 # a = self.vocab.numberize(a)
#                 count[a] += 1
#                 total += 1
#         self.logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
#                                              for a in self.vocab}

#     def start(self) -> Sequence[str]:
#         """Return the language model's start state."""
        
#         return (self.n - 1) * (START_TOKEN)

#     def step(self,
#              q: Sequence[str],
#              w: str
#              ) -> Tuple[Sequence[str], Mapping[str, float]]:
#         """Compute one step of the language model.

#         Arguments:
#         - q: The current state of the model
#         - w: The most recently seen token (str)

#         Return: (r, pb), where
#         - r: The state of the model after reading `w`
#         - pb: The log-probability distribution over the next token
#         """

#         # delete first string and add new token
#         r = q[1:] + [w] # type: ignore

#         pb = self.logprobs.get(r, {v: math.log(1/(len(self.vocab)-1)) for v in self.vocab if v != utils.START_TOKEN})

#         return (r, pb)
    
# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import Counter, defaultdict
from typing import Tuple
import math


# PYTHON PROJECT IMPORTS
from utils import START_TOKEN, END_TOKEN, Vocab


class Uniform(object):
    def __init__(self, vocab):
        self.vocab = vocab

        self.uniform_dict = {v: math.log(1/(len(self.vocab)-1)) for v in self.vocab if v != START_TOKEN}
        self.logprobs = defaultdict(lambda: self.uniform_dict)


class Ngram(object):
    def __init__(self,
                 n: int,
                 data: Sequence[Sequence[str]]):
        self.n = n
        self.vocab = Vocab()

        # print(f"training {n}-gram model")

        # collect some counts
        gram_to_counts = defaultdict(lambda: defaultdict(int))
        for seq in data:
            gram = self.start()

            if len(seq) > 0 and seq[0] == START_TOKEN:
                seq = seq[1:]
            if len(seq) > 0 and seq[-1] != END_TOKEN:
                seq = seq + [END_TOKEN]

            for c in seq:
                self.vocab.add(c)
                gram_to_counts[gram][c] += 1
                gram = gram[1:] + (c,)

        if self.n > 0:
            self.smoothing_model = Ngram(self.n-1, data)


            # count the number of unique tokens seen at least 1 time per gram
            gram_to_num_tokens_seen_at_least_once = defaultdict(int)
            gram_to_num_tokens_seen_at_least_twice = defaultdict(int)
            for gram, counts in gram_to_counts.items():
                gram_to_num_tokens_seen_at_least_once[gram] = len(counts)
                for _, count in counts.items():
                    if count >= 2:
                        gram_to_num_tokens_seen_at_least_once[gram] += 1

            # print(f"smoothing {n}-gram with {n-1}-gram")

            # convert to logprobs
            self.logprobs = defaultdict(dict)
            for gram, counts in gram_to_counts.items():
                total = sum(counts.values())

                lam = total / (total + gram_to_num_tokens_seen_at_least_once[gram])

                for v in self.vocab:
                    if v != START_TOKEN:
                        pr_v_given_gram = counts[v] / total

                        self.logprobs[gram][v] = math.log(lam * pr_v_given_gram +
                            (1 - lam)*math.exp(self.smoothing_model.logprobs[gram[1:]][v]))
        else:
            self.smoothing_model = Uniform(self.vocab)
            self.uniform_dict = {v: math.log(1/(len(self.vocab)-1)) for v in self.vocab if v != START_TOKEN}
            self.logprobs = defaultdict(lambda: self.uniform_dict)

    def start(self) -> Sequence[str]:
        return (START_TOKEN,) * (self.n-1)

    def step(self,
             q: Sequence[str],
             w: str) -> Tuple[Sequence[str], Mapping[str, float]]:
        # append w and delete old first token
        r = q[1:] + (w,)

        p = self.logprobs.get(r, {v: math.log(1/(len(self.vocab)-1)) for v in self.vocab if v != START_TOKEN})

        return r, p

