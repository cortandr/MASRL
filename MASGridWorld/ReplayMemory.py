import random


class ReplayMemory:

    def __init__(self, table_size=200000, batch_size=32):
        self.memory = list()
        self.table_size = table_size
        self.batch_size = batch_size

    def insert(self, transition):
        self.memory.append(transition)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def get_size(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.table_size

    def refresh(self):
        prune = self.table_size // 5
        self.memory = self.memory[prune:]

    def can_replay(self):
        """
        Define if the memory has enough data to start replay training
        :return: bool
        """
        return len(self.memory) > self.batch_size
