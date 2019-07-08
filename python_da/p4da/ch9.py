import numpy as np
import matplotlib.pyplot as plt

data = np.arange(10)
plt.plot(data)

fig = plt.figure()  # create figure object
ax1 = fig.add_subplot(2, 2, 1)  # add 2 x 2 subplots to fig, initialize at pos 1
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

plt.plot([1.5, 3.5, -2, 1.6])  # draw on last active plot
plt.plot(np.random.randn(50).cumsum(), 'k--')  # note how axis are changed automatically
# k ... style option, --: use dashed line
# plot to a subplot by calling its axis
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

fig, axes = plt.subplots(2, 3)  # convenience function to create subplots
type(axes)  # note: is numpy ndarray and can be indexed [, ] style
axes.shape  # and has shape (2, 3)
axes[1, 0] scatter(np.arange(20), np.arange(20) + 2 * np.random.randn(20))

# plot parameters
x = np.random.randn(30)
y = 5 + 0.8 * x + np.random.randn(30)
plt.plot(x, y, linestyle = "--", color = "g")
plt.close()
plt.scatter(x, y, marker = "o", color = "#b2df8a")  # accepts hex
plt.scatter(x, y, label = "Random Data")
plt.close()

data = np.random.randn(50).cumsum()
plt.plot(data)  # line charts are linearly interpolated by default
plt.plot(data, "k--", label="Default")
plt.plot(data, "k-", drawstyle="steps-post", label="steps-post")
plt.legend(loc="best")  # required to create legend, will work even if no labels set
