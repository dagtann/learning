# k-Nearest Neighbors
import requests
from collections import Counter, defaultdict
import csv
from matplotlib import pyplot as plt
from scratch.linear_algebra import Vector, distance
from typing import List, NamedTuple, Dict


def majority_vote(labels: List[str]) -> str:
    """Assumes labels are ordered from closest to fartherst"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    # extracts most freq from list
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # try again dropping last label


assert majority_vote(["a", "b", "c", "b", "a"]) == "b"
# tiest on first run, then removes be, so returns b


# create knn classifier
class labeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[labeledPoint],
                 new_point: Vector) -> str:
    # Order labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point, new_point))

    # Find the labels for the k closest points
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # take a vote
    return majority_vote(k_nearest_labels)


# Demonstration with iris data
data = requests.get(
                   "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "iris/iris.data")
with open("iris.data", "w") as f:
    f.write(data.text)


def parse_iris_row(row: List[str]) -> labeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1]]
    # class is e.g. "Iris-virginica", drop iris part
    label = row[-1].split("-")[-1]
    return labeledPoint(measurements, label)


with open("iris.data", "r") as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader]

# Exploratory analysis
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

metrics = ["sepal length", "sepal width", "petal length", "petal width"]
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
fig, ax = plt.subplots(2, 3)
marks = ["+", ".", "x"]

for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize = 8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc="lower right", prop={"size": 6})
plt.show()

import random
from scratch.machine_learning import split_data
random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.7)

from typing import Tuple

# track how many times we see (predicted, actual)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)