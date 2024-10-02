# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from data.charloader import load_chars_from_file


# PYTHON PROJECT IMPORTS
from charpredictor import CharPredictor
from utils import read_mono

def train_model():
    return CharPredictor(2, "data/mandarin/charmap", "data/mandarin/train.han")

def dev_model(model):
    dev_pin = load_chars_from_file("data/mandarin/dev.pin")
    dev_han = load_chars_from_file("data/mandarin/dev.han")

    # remember: the errors down below are expected (I think)

    num_correct: int = 0
    total: int = 0

    i = 0
    for pin, han in zip(dev_pin, dev_han):
        if (i % 1000 == 0):
            print(f"p/h   {pin} - {han}")
        i += 1

        q = model.start()

        for c_input, c_actual in zip(pin[1:-1], han[1:-1]):
            q, p = model.step(q, c_input)
            q, _ = model.model.step(q, c_actual)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            num_correct += int(c_predicted == c_actual)
            total += 1

    return num_correct, total

def test_model(m):
    pin_data = read_mono("./data/mandarin/test.pin", delim=" ")
    han_data: Sequence[Sequence[str]] = read_mono("./data/mandarin/test.han", delim='')

    num_correct: int = 0
    total: int = 0
    for pin_line, han_line in zip(pin_data, han_data):

        q = m.start()

        for c_input, c_actual in zip(pin_line[1:-1], han_line[1:-1]):
            q, p = m.step(q, c_input)
            q, _ = m.lm.step(q, c_actual)

            c_predicted = max(p.keys(), key=lambda k: p[k])

            num_correct += int(c_predicted == c_actual)
            total += 1

    return num_correct, total

if __name__ == "__main__":
    my_model = train_model()
    dev_model(my_model)