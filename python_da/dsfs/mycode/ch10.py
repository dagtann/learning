from numpy.random import binomial
import pandas as pd

P_LUKE = 0.005
P_LEUKEMIA = 0.014
N = int(1e6)

df = pd.DataFrame({"lukes": [binomial(1, P_LUKE) for _ in range(N)],
                   "leukemia": [binomial(1, P_LEUKEMIA) for _ in range(N)]})
pd.crosstab(df["lukes"], df["leukemia"], margins=True)


def fit_scores(tp=None, fp=None, tn=None, fn=None):
    accuracy = (tp + tn) / sum([tp, fp, tn, fn])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\n" +
          f"F1: {f1}")
    return None


fit_scores(tp=60, fp=5013, tn=981017, fn=13910)