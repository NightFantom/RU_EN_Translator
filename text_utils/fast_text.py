import numpy as np
from typing import List


class FastTextWrapper:

    def __init__(self, embedder):
        self.embedder = embedder

    def transform(self, data: List[List[str]]) -> List[np.array]:
        res = []
        for sentence in data:
            temp = []
            for x in sentence:
                if x in self.embedder.wv:
                    temp.append(x)
            if len(temp) != 0:
                vector_np = self.embedder.wv[temp]
                vector_np = vector_np.astype(np.float32)
            else:
                vector_np = np.zeros((1, self.embedder.vector_size), dtype=np.float32)
            res.append(vector_np)
        return res