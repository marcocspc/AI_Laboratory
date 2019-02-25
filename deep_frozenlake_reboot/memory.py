import random

#peguei este codigo do tutorial: 
# https://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/
#o codigo de la eh bem confuso, mas esta classe de memoria esta bem organizada
#o codigo da classe do agente tb foi adaptado de la, mas houve bastante modificacao

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

