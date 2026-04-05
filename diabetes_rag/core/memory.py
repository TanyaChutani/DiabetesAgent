class Memory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add(self, query, response):
        self.history.append((query, response))
        self.history = self.history[-self.max_turns:]

    def get(self):
        return self.history
