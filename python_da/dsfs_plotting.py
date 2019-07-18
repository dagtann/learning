from collections import Counter
from matplotlib import pyplot as pyplot

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
histogram = Counter([min(grade // 10 * 10, 90) for grade in grades])
# buckets by decile, but 100 grouped with 90

plt.bar([x + 5 for x in histogram.keys()], histogram.values(), 10, edgecolor = [1, 1, 1])
plt.axis([-5, 105, 0, 5])

plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

# line chart
variance = [2 ** i for i in range(0, 9)]
bias_squared = [variance[-i] for i in range(1, len(variance) + 1)]
total_error = [b + v for b, v in zip(bias_squared, variance)]
xs = [i for i, _ in enumerate(variance)]

plt.plot(xs, variance, "g-", label = "variance")
plt.plot(xs, bias_squared, "r-", label = "bias^2")
plt.plot(xs, total_error, "b:", label = "total error")

plt.legend(loc=9)
plt.xlabel("Model complexity")
plt.xticks([])
plt.title("The Bias-Vaiance-Tradeoff")
plt.show()