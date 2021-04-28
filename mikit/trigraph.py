import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors
import os
import pandas as pd
import numpy as np


class TriGraph:
    def  __init__(self, atoms):
        """
        Args:
            origin_tar_atoms(list) : List of target atoms
        """
        # Set figure
        fig = plt.figure()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.xlim(-0.15,1.15)
        plt.ylim(-0.15,1)
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', 'datalim')
        ax1.tick_params(labelbottom = False, labelleft = False, labelright = False, labeltop = False)
        ax1.tick_params(bottom = False, left = False, right = False, top = False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        h = np.sqrt(3.0)*0.5

        # outer circumference
        ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw = 2)
        ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw = 2)
        ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw = 2)

        # Vertex labels
        ax1.text(0.5, h+0.1, atoms[0], fontsize = 16, ha = 'center', va = 'top')
        ax1.text(-0.02 - len(atoms[1]) * 0.05, -0.02, atoms[1], fontsize = 16)
        ax1.text(1.025, -0.02, atoms[2], fontsize = 16, zorder=10, linespacing = 0)

        # Axis label
        for i in range(1,10):
            ax1.text(0.52+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            ax1.text((10-i)/20.0-0.07, h*(10-i)/10.0-0.0, '%d0' % i, fontsize=10)
            ax1.text(i/10.0-0.03, -0.06, '%d0' % i, fontsize=10)

        # Inner scale
        for i in range(1,10):
            ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='#AAAAAA', lw=0.5, zorder=2)

        self.fig = fig
        self.ax1 = ax1
        self.c_set = False

    def get_graph(self):
        return self.fig
    
    @staticmethod
    def _get_xy(tensor):
        """
        Args:
            tensor(numpy) : Explanatory variables {tensor(x, y, z)}
        Return:
            view_x(numpy) : X-coordinates in a triangle diagram
            view_y(numpy) : Y-coordinates in a triangle diagram
        """
        x, y, z = tensor[:,0], tensor[:,1], tensor[:,2]
        view_x = 0.5 * (x + 2 * z) / (x + y + z)
        view_y = 3**0.5 *0.5 * (x) / (x + y + z)
        return view_x, view_y
    
    def set_colorbar(self, view_z):
        """
        Args:
            view_z(numpy) : Objective variable
        """
        if not self.c_set:
            # Color bar range
            vmin = np.min(view_z)
            vmax = np.max(view_z)
            self.norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)

            # Tint
            self.levels = []
            if vmax > 0:
                lim = vmax * 51 / 50
            else:
                lim = vmax * 49 / 50

            while vmin <= lim:
                self.levels.append(float(vmin))
                vmin = float(vmin) + abs(float(vmax)) / 50
            self.cmap = plt.cm.rainbow

            # Set the color bar
            self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            self.sm.set_array([])
            self.fig.colorbar(self.sm, shrink=0.8)
            self.c_set = True


    def add_contourf(self, tensor, view_z):
        """
        Args:
            tensor(numpy) : Explanatory variables {tensor(x, y, z)}
            view_z(numpy) : Objective variable
        """
        view_x, view_y = self._get_xy(tensor)
        T = tri.Triangulation(view_x, view_y)
        self.set_colorbar(view_z)
        self.ax1.tricontourf(view_x, view_y, T.triangles, view_z, cmap = self.cmap, norm = self.norm, levels = self.levels, zorder=1)
        plt.rcParams['font.family'] = 'Times New Roman'
    
    def add_plot(self, tensor, view_z=np.array([None]), marktype=None):
        """
        Args:
            tensor(numpy) : Explanatory variables {tensor(x, y, z)}
            view_z(numpy) : Objective variable
            marktype(bool, str) : Type of the marker
        """
        view_x, view_y = self._get_xy(tensor)
        if not np.any(view_z) and marktype == None:
            self.ax1.scatter(view_x, view_y, c = "black", s = 40, linewidth = 1, zorder=4)
        elif marktype == "b":
            self.ax1.scatter(view_x, view_y, c = "yellow", s = 200, linewidth = 1, zorder=4, marker = "*", edgecolor = 'black')
        else:
            T = tri.Triangulation(view_x, view_y)
            self.set_colorbar(view_z)
            self.ax1.scatter(view_x[::-1], view_y[::-1], c = view_z[::-1], s = 40, linewidth = 1, edgecolor = 'black', norm = self.norm, cmap = self.cmap, zorder=4)
