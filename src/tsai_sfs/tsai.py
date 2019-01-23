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

def sfs(filepath, N_iter=100):
    img = cv2.imread(filepath)
    grayscale = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.float)
    # normalize
    grayscale_max = grayscale.max()
    E = grayscale / grayscale_max
    albedo, slant, tilt, L = estimate_albedo(E)

    Zn = np.zeros(E.shape, dtype=np.float)
    Si = np.zeros(E.shape, dtype=np.float) + 0.01
    wn = 1e-8

    ps = math.cos(tilt) * math.sin(slant) / math.cos(slant)
    qs = math.sin(tilt) * math.sin(slant) / math.cos(slant)
    pqs = 1.0 + ps ** 2 + qs ** 2

    for iter in range(N_iter):
        P, Q = np.gradient(Zn)
        PQ = 1.0 + P ** 2 + Q ** 2
        Ei_tmp = (1 + P * ps + Q * qs) / (np.sqrt(PQ) + np.sqrt(pqs))
        Ei = np.where(Ei_tmp < 0, 0, Ei_tmp)
        fZ = -1.0 * (E - Ei)
        dfZ = -1.0 * ((ps + qs)/(np.sqrt(PQ) + np.sqrt(pqs)) 
                      - (P + Q) * (1.0 + P*ps + Q*qs) / (np.sqrt(PQ ** 3) * np.sqrt(pqs)))
        Y = fZ + dfZ * Zn
        K = Si * dfZ / (wn + dfZ * Si * dfZ)
        
        Si = (1.0 - K * dfZ) * Si
        Zn = Zn + K * (Y - dfZ * Zn)
    return Zn

if __name__ == "__main__":
    default_filepath = "../../img/car_camera.jpg"
    Z = sfs(default_filepath)
    xx, yy = np.meshgrid(np.arange(1, Z.shape[1]+1), np.arange(1, Z.shape[0]+1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, Z, cmap="bwr")
    plt.show()
    fig.show()