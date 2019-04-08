class FastTextWrapper:

    def __init__(self, embedder):
        self.embedder = embedder

    def transform(self, data):
        res = []
        for sentence in data:
            temp = []
            for x in sentence:
                if x in self.embedder.wv:
                    temp.append(x)
            vector_np = self.embedder.wv[temp]
            res.append(vector_np)
        return res