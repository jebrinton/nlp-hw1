# use absolute discounting (bottom right side of slide 14 of lecture 4)

from collections.abc import Sequence
from data.charloader import load_chars_from_file
from ngram import Ngram
from utils import Vocab
import utils

def train_ngram(n):
    path = "data/english/train"
    data = load_chars_from_file(path)
    model = Ngram(n, data)
    return model

def dev_ngram(model):
    dev_path = "data/english/dev"
    dev_data = load_chars_from_file(dev_path)

    # remember: the errors down below are expected (I think)

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = model.start()
        for c_input, c_actual in zip([utils.START_TOKEN] + dev_line, dev_line + [utils.END_TOKEN]):
            q, p = model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            num_correct += int(c_predicted == c_actual)
            total += 1

    print(num_correct / total)

if __name__ == "__main__":
    my_model = train_ngram(5)
    dev_ngram(my_model)