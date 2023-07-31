from scipy.spatial.distance import cosine
import numpy as np


class OpenAPIFunction:
    def __init__(self, word_from, word_to, demo=True, pos_word=None):
        self.from_text = word_from
        self.to_text = word_to
        self.switch = 1
        self.demo = demo
        if word_from.text.startswith('Non-'):
            print('triggered negation', word_from.text, pos_word.text)
            self.switch = -1
            self.from_text = pos_word

        if demo:
            list = np.random.random((2, 1))
            neg = list[1] < 0.5
            value = float(list[0])
            self.distance = (1 + (-2 * neg)) * value
        else:
            self.distance = cosine(self.from_text.embedding, self.to_text.embedding)

    def calculate(self, value):
        print('from word', self.from_text.text, 'to word', self.to_text.text, 'distance', self.distance, 'val', value)
        return self.switch * value * (1 - self.distance)
