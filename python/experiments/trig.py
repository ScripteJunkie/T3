from locale import getpreferredencoding
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

# p = [[0.571, 2.577], [6.269, 5.121], [10.267, 4.241], [5, 1.45]]
# p1 = [[-1, -1], [3, 3], [11, 2], [7, -2]]
# p1 = [[603, 54], [1478, 558], [1043, 726], [147, 471]]
p1 = [[141,637], [986,120], [1446,491], [631,779]]
p2 = [[913,47], [1755,512], [1281,670], [477,426]]
# p2 = [[-703, 94], [-1478, 558], [-1093, 626], [-147, 471]]
# p2 = [[-800, -500], [-200, 200], [200, 200], [800, -500]]
p1Fin = [[-2.5, -4.5], [-2.5, 4.5], [2.5, 4.5], [2.5, -4.5]]
p2Fin = [[-2.5, -4.5], [-2.5, 4.5], [2.5, 4.5], [2.5, -4.5]]
p1Tran = []
p2Tran = []
pChg = []
pChg2 = []

def getPerspectiveTransform(sourcePoints, destinationPoints):
    """
    Calculates the 3x3 matrix to transform the four source points to the four destination points

    Comment copied from OpenCV:
    /* Calculates coefficients of perspective transformation
    * which maps soruce (xi,yi) to destination (ui,vi), (i=1,2,3,4):
    *
    *      c00*xi + c01*yi + c02
    * ui = ---------------------
    *      c20*xi + c21*yi + c22
    *
    *      c10*xi + c11*yi + c12
    * vi = ---------------------
    *      c20*xi + c21*yi + c22
    *
    * Coefficients are calculated by solving linear system:
    *             a                         x    b
    * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
    * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
    *
    * where:
    *   cij - matrix coefficients, c22 = 1
    */

    """
    if sourcePoints.shape != (4,2) or destinationPoints.shape != (4,2):
        raise ValueError("There must be four source points and four destination points")

    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]

    x = np.linalg.solve(a, b)
    x.resize((9,), refcheck=False)
    x[8] = 1 # Set c22 to 1 as indicated in comment above
    return x.reshape((3,3))



# Finds equation of line between points -> list of points, index1, index2, pass x
def line(list, point1, point2, x):
    index = point1-1
    index2 = point2-1
    m = (list[index2][1]-list[index][1])/(list[index2][0]-list[index][0])
    y = (m*(x))-(m*list[index][0])+(list[index][1])
    return y


# Very wrong?
def eqtn(list, point1, point2, point3, point4):
    # Expects 
    # 1 ----- 4
    # |       |
    # |       |
    # 2 ----- 3
    index = point1-1
    index2 = point2-1
    index3 = point3-1
    index4 = point4-1
    # gets slopes of horizontal and vertical table edges.
    m1_1 = (list[index2][1]-list[index][1])/(list[index2][0]-list[index][0])
    m1_2 = (list[index4][1]-list[index3][1])/(list[index4][0]-list[index3][0])
    c1_1 = ((-list[index][0])*m1_1)+list[index][1]
    c1_2 = ((-list[index3][0])*m1_2)+list[index3][1]
    # print(c1_1, c1_2)
    if m1_1 == m1_2:
        x1 = -(c1_1 - c1_2)
    else:
        x1 = -(c1_1 - c1_2) / (m1_1 - m1_2)
    y1 = (m1_1 * x1) + c1_1 
    # print(x1, y1)
    m2_1 = (list[index4][1]-list[index][1])/(list[index4][0]-list[index][0])
    m2_2 = (list[index3][1]-list[index2][1])/(list[index3][0]-list[index2][0])
    c2_1 = ((-list[index][0])*m2_1)+list[index][1]
    c2_2 = ((-list[index3][0])*m2_2)+list[index3][1]
    if m2_1 == m2_2:
        x2 = -(c2_1 - c2_2)
    else:
        x2 = -(c2_1 - c2_2) / (m2_1 - m2_2) 
    y2 = (m2_1 * x2) + c2_1 
    # print(x2, y2)
    # print(c2_1, c2_2)
    # y1 = (m1*(x))-(m1*list[index][0])+(list[index][1])
    # y2 = (m2*(x))-(m2*list[index][0])+(list[index][1])
    d = math.sqrt((x2-x1)**2+(y2-y1)**2)
    FOVx = 90.0* 1920 / d
    print("X FOV: ", FOVx)
    return [x1, x2], [y1, y2]

# Transforms x, y, z points using M transform matrix
def pTransform(M, x, y, z):
    F = np.reshape(np.array((x, y, z)), (3,1))
    # T = (M[0,0]*F[0] + M[0,1]*F[1] + M[0,2]*F[2]), (M[1,0]*F[0] + M[1,1]*F[1] + M[1,2]*F[2]), (M[2,0]*F[0,0] + M[2,1]*F[1] + M[2,2]*F[2])
    T = np.matmul(M, F)
    return T

def pCompute(currP, tarP):
    # initail lis tof points (4), target points (4)
    A = []
    for p in range(len(currP)):
        x = [currP[p][0],currP[p][1],1,0,0,0, -currP[p][0]*tarP[p][0],-tarP[p][0]*currP[p][1],-tarP[p][0]]
        y = [0,0,0,currP[p][0],currP[p][1],1, -currP[p][0]*tarP[p][1],-tarP[p][1]*currP[p][1],-tarP[p][1]]
        A.append(x)
        A.append(y)
        # print(x)
        # print(y)
    u, s, v = np.linalg.svd(A)
    h = v
    # print(v)
    H = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            H[i,j]= h[i+j][0]
    H[2,2] = 1.0
    return H

# x parameters for plotting -> xmin, xman, steps
x = np.linspace(-2500, 2500, 1000)

# Center of points
center = (p1[0][0]+p1[1][0]+p1[2][0]+p1[3][0])/4, (p1[0][1]+p1[1][1]+p1[2][1]+p1[3][1])/4

# Test point
# test1 = [11, 7]
test1 = [428,518]
# Translated around center
test1 = [test1[0]-center[0], test1[1]-center[1]]

# Camera 1
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(figsize=(5, 5))
plt.xlim(-1500, 1500)
plt.ylim(-1500, 1500)
# Plot points
for i in range(len(p1)):
    # plt.plot(p1[i][0], p1[i][1], marker="o", markersize=3, markeredgecolor="magenta", markerfacecolor="magenta")
    p1Tran.append(((p1[i][0]-center[0]), (p1[i][1]-center[1])))
    plt.plot(p1Tran[i][0], p1Tran[i][1], marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan")
# Plot lines
plt.plot(x, line(p1Tran, 1, 2, x), color="yellow", linewidth="0.5")
plt.plot(x, line(p1Tran, 2, 3, x), color="white", linewidth="0.5")
plt.plot(x, line(p1Tran, 3, 4, x), color="white", linewidth="0.5")
plt.plot(x, line(p1Tran, 1, 4, x), color="white", linewidth="0.5")
# Plot center of points
# plt.plot(center[0], center[1], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
# Plot transformed center of points/center frame
plt.plot(0, 0, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
# Plot transformed center of points (test2)
plt.plot(test1[0], test1[1], marker=".", markersize=3, markeredgecolor="yellow", markerfacecolor="yellow")

l = eqtn(p1Tran, 1, 2, 3, 4)
plt.plot(l[0], l[1], color="green", linewidth="1", linestyle="dotted")

plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.title("Camera View")

# Generates perspective tranform between current points and target points
M1 = cv2.getPerspectiveTransform(np.float32(p1Tran),np.float32(p1Fin))
# M1Test = pCompute(np.float32(p1Tran),np.float32(p1Fin))
M1Test = getPerspectiveTransform(np.float32(p1Tran),np.float32(p1Fin))

print("M1: ", type(M1), '\n', M1.shape, '\n', M1, '\n\n', "Test: ", type(M1Test), '\n', M1Test.shape, '\n', M1Test)

center2 = (p2[0][0]+p2[1][0]+p2[2][0]+p2[3][0])/4, (p2[0][1]+p2[1][1]+p2[2][1]+p2[3][1])/4

# test2 = [1, 8]
test2 = [1453, 402]
test2 = [test2[0]-center2[0], test2[1]-center2[1]]

# Camera 2
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(figsize=(5, 5))
plt.xlim(-1500, 1500)
plt.ylim(-1500, 1500)
# Plot points
for i in range(len(p1)):
    # plt.plot(p2[i][0], p2[i][1], marker="o", markersize=3, markeredgecolor="magenta", markerfacecolor="magenta")
    p2Tran.append(((p2[i][0]-center2[0]), (p2[i][1]-center2[1])))
    plt.plot(p2Tran[i][0], p2Tran[i][1], marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan")
# Plot lines
plt.plot(x, line(p2Tran, 1, 2, x), color="yellow", linewidth="0.5")
plt.plot(x, line(p2Tran, 2, 3, x), color="white", linewidth="0.5")
plt.plot(x, line(p2Tran, 3, 4, x), color="white", linewidth="0.5")
plt.plot(x, line(p2Tran, 1, 4, x), color="white", linewidth="0.5")
# Plot center of points
# plt.plot(center2[0], center2[1], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
# Plot transformed center of points/center frame
plt.plot(0, 0, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
# Plot transformed center of points (test2)
plt.plot(test2[0], test2[1], marker=".", markersize=3, markeredgecolor="yellow", markerfacecolor="yellow")

l = eqtn(p2Tran, 1, 2, 3, 4)
plt.plot(l[0], l[1], color="green", linewidth="1", linestyle="dotted")

plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.title("Camera 2 View")

# Generates perspective tranform between current points and target points
M2 = cv2.getPerspectiveTransform(np.float32(p2Tran),np.float32(p2Fin))
# print(M2)

# Top View
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(3, figsize=(5, 5))
plt.xlim(-15, 15)
plt.ylim(-15, 15)


# Since both transform matricies (M1, M2) should be correct we olny need one set of transformed corners to be shown
for m in range(len(p1Tran)):
    # print(M1Test.shape, M1.shape)
    T1 = pTransform(M1Test, p1Tran[m][0], p1Tran[m][1], 0)
    T2 = pTransform(M2, p2Tran[m][0], p2Tran[m][1], 0)
    x1 = p1Fin[m][0]
    y1 = 1
    pChg.append((T1[0], T1[1]))
    pChg2.append((T2[0], T2[1]))

# Top View lines drawn
# plt.plot(x, line(pChg, 1, 2, x)) # Isn't function when in top view
plt.plot(x, line(pChg, 2, 3, x), color="white", linewidth="0.5")
# plt.plot(x, line(pChg, 3, 4, x)) # Isn't function when in top view
plt.plot(x, line(pChg, 1, 4, x), color="white", linewidth="0.5")

# plt.plot(x, line(pChg2, 1, 2, x)) # Isn't function when in top view
plt.plot(x, line(pChg2, 2, 3, x), color="white", linewidth="0.5")
# plt.plot(x, line(pChg2, 3, 4, x)) # Isn't function when in top view
plt.plot(x, line(pChg2, 1, 4, x), color="white", linewidth="0.5")

# On a fully transformed top view there are only vertical and horizontal lines, so need to plot x=? with a vertical line.
plt.axvline(x=-2.5, color="yellow", linewidth="0.5")
plt.axvline(x=2.5, color="white", linewidth="0.5")
for m in range(len(p1Tran)):
    # print(pChg[m])
    plt.plot(pChg[m][0], pChg[m][1], marker="o", markersize=4, markeredgecolor="red", markerfacecolor="cyan")
    plt.plot(pChg2[m][0], pChg2[m][1], marker="o", markersize=4, markeredgecolor="blue", markerfacecolor="green")

# Transforms points (test1) from Camera 1 to Top View
T1 = pTransform(M1, test1[0], test1[1], 0)
plt.plot(T1[0], T1[1], marker=".", markersize=4, markeredgecolor="yellow", markerfacecolor="yellow")

# Transforms points (test2) from Camera 2 to Top View
T2 = pTransform(M2, test2[0], test2[1], 0)
plt.plot(T2[0], T2[1], marker=".", markersize=4, markeredgecolor="yellow", markerfacecolor="yellow")

# Trying to find optical axis line (Not correct)
plt.plot(x, line([[0, 0], [-center[0], pTransform(M1, 0.0, test1[1]+center[1], 0.0)[1]]], 1, 2, x), color="blue", linewidth="0.5")
plt.plot(x, line([[0, 0], [-center2[0], pTransform(M2, 0.0, test2[1]+center2[1], 0.0)[1]]], 1, 2, x), color="green", linewidth="0.5")
# plt.plot(x, line([[0, 0], [T2[0], T2[1]]], 1, 2, x), color="green", linewidth="0.5")


# This section generates the transform matrix (M3) to take points (test2) from Camera 2 and translate them to Camera 1's
# perspective (T3) then to prove its fuctionality it does a tranform with the matrix (M1) used to translate
# Camera 1 into the Top View. This then leaves us with a point fully translated to any perspective needed.
# M3 = cv2.getPerspectiveTransform(np.float32(p2Tran),np.float32(p1Tran))
# T3 = pTransform(M3, test2[0], test2[1], 0)
# T4 = pTransform(M1, T3[0], T3[1], 0)
# plt.plot(T4[0], T4[1], marker="o", markersize=4, markeredgecolor="yellow", markerfacecolor="cyan")

plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.title("Top View")

# Show legend
# plt.legend()
# Show graphs
plt.show()