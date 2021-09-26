class Buffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.elements = []

    def get_first_element(self):
        return self.elements.pop(0)

    def get_element(self, idx):
        return self.elements.pop(idx)

    def get_current_len(self):
        return len(self.elements)

    def __len__(self):
        return len(self.elements)

    def add_element(self, element):
        assert self.get_current_len() < self.max_len, "Exceeded max buffer length."
        self.elements.append(element)
        return

    def __repr__(self):
        return str(self.elements)
