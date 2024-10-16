import numpy as np

from artlib.experimental.SeqART import SeqART


sequences = [
    "hello",
    "helsinki",
    "hella",
    "hell",
    "arkansas",
    "kansas",
    "advent",
    "adventure",
    "ventilation",
    "addition",
    "subtraction",
    "substitution",
]

def cluster_sequences():
    X = np.array(sequences, dtype=object)

    params = {
        "rho": -0.5
    }
    cls = SeqART(**params)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    for i in range(cls.n_clusters):
        print(f"Cluster {i}")
        for s_i in range(len(sequences)):
            if i == y[s_i]:
                print(sequences[s_i])
        print(cls.W[i])
        print("=" * 20)


if __name__ == "__main__":
    cluster_sequences()


