import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class CubeVisualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

    def update_visualization(self, supernodes):
        """Update the Cube visualization based on SuperNodes."""
        self.ax.clear()
        for node in supernodes:
            x, y, z = node["coordinates"]
            self.ax.scatter(x, y, z, c="b", marker="o")
        plt.draw()
        plt.pause(0.5)

    def show(self):
        plt.show()
