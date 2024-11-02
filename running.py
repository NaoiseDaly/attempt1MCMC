from functions import *
import matplotlib.pyplot as plt
first_chain = basic_metropolis_hastings(-5, 600)

fig, ax = plt.subplots()
ax.plot(first_chain, "-")
ax.set_xlabel("t")
ax.set_ylabel("X")
plt.show()