from collections.abc import Sequence
from data.charloader import load_chars_from_file
from unigram import Unigram
from utils import Vocab
import utils

def train_unigram():
    path = "data/english/train"
    data = load_chars_from_file(path)
    model = Unigram(data)
    return model

def dev_unigram(uni_model):
    dev_path = "data/english/dev"
    dev_data = load_chars_from_file(dev_path)

    # remember: the errors down below are expected (I think)

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = uni_model.start()
        for c_input, c_actual in zip([utils.START_TOKEN] + dev_line, dev_line + [utils.END_TOKEN]):
            q, p = uni_model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            num_correct += int(c_predicted == c_actual)
            total += 1

    print(num_correct / total)

if __name__ == "__main__":
    my_model = train_unigram()
    dev_unigram(my_model)