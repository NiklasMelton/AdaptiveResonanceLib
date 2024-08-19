import numpy as np
from artlib import GramART


sentences = [
    "The cat sleeps on the mat.",
    "The dog sleeps on the couch.",
    "The child plays in the park.",
    "The cat plays with the toy.",
    "The dog barks at the mailman.",
    "The bird sings in the morning.",
    "The child sleeps in the crib.",
    "The bird flies in the sky."
]

text_data = [
    s.split()
    for s in sentences
]

def cluster_sentences():
    X = np.array(sentences, dtype=object)

    params = {
        "rho": 0.5
    }
    cls = GramART(**params)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    for i in range(cls.n_clusters):
        print(f"Cluster {i}")
        for s_i in range(len(sentences)):
            if i == y[s_i]:
                print(sentences[s_i])
        print("="*20)

if __name__ == "__main__":
    cluster_sentences()