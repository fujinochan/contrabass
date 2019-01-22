#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math

import cv2
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D


def estimate_albedo(E):
    eps = 1e-10
    m1 = np.average(E)
    m2 = np.average(E ** 2)
    Ex, Ey = np.gradient(E)
    Exy = np.sqrt(Ex**2 + Ey**2) + eps
    nEx = Ex / Exy
    nEy = Ey / Exy
    avgEx = np.average(nEx)
    avgEy = np.average(nEy)

    gamma = np.sqrt((6. * (math.pi**2) * m2) - (48. * (m1**2)))
    albedo = gamma / math.pi
    slant = math.acos((4.*m1) / gamma)
    tilt = math.atan(avgEy/avgEx)
    if tilt < 0:
        tilt = tilt + math.pi
    L = (math.cos(tilt) * math.sin(slant),
         math.sin(tilt) * math.sin(slant),
         math.cos(slant))
    return albedo, slant, tilt, L


def pentland(filepath):
    img = cv2.imread(filepath)
    grayscale = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.float)
    # normalize
    grayscale_max = grayscale.max()
    E = grayscale / grayscale_max
    albedo, slant, tilt, L = estimate_albedo(E)
    Fe = np.fft.fft2(E)

    xx, yy = np.meshgrid(np.arange(1, E.shape[1]+1), np.arange(1, E.shape[0]+1))
    wx = (2.*math.pi*xx) / E.shape[0]
    wy = (2.*math.pi*yy) / E.shape[1]

    Fz = Fe / (-1j * wx * math.cos(tilt) * math.sin(slant) - 1j * wy * math.sin(tilt) * math.cos(slant))

    Z = np.abs(np.fft.ifft2(Fz))
    return Z


if __name__ == "__main__":
    default_filepath = "../../img/Nike.jpg"
    Z = pentland(default_filepath)
    xx, yy = np.meshgrid(np.arange(1, Z.shape[1]+1), np.arange(1, Z.shape[0]+1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, Z)
    plt.show()
    fig.show()