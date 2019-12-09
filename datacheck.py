from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

c1 = np.load('camMtx1.npy')
c2 = np.load('camMtx2.npy')

d1 = np.load('distCoeffs1.npy')
d2 = np.load('distCoeffs2.npy')
T = np.load('Transl.npy')
R1 = np.load('R1.npy')
P1 = np.load('P1.npy')
R2 = np.load('R2.npy')
P2 = np.load('P2.npy')
Q = np.load('Q.npy')

print('c1: ', c1)

print('c2: ', c2)
print('d1: ', d1)
print('d2: ', d2)
print('T: ', T)
print('R1: ', R1)
print('P1: ', P1)
print('R2: ', R2)
print('P2: ', P2)
print('Q: ', Q)
