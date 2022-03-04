from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import cv2

# p = [[0.571, 2.577], [6.269, 5.121], [10.267, 4.241], [5, 1.45]]
p1 = [[-1, -1], [3, 3], [11, 2], [7, -2]]
p2 = [[3, -5], [-7, 0], [-2, 2], [8, -3]]
p1Fin = [[-2.5, -4.5], [-2.5, 4.5], [2.5, 4.5], [2.5, -4.5]]
p2Fin = [[-2.5, -4.5], [-2.5, 4.5], [2.5, 4.5], [2.5, -4.5]]
p1Tran = []
p2Tran = []
pChg = []
pChg2 = []



# Finds equation of line between points -> list of points, index1, index2, pass x
def line(list, point1, point2, x):
    index = point1-1
    index2 = point2-1
    m = (list[index2][1]-list[index][1])/(list[index2][0]-list[index][0])
    y = (m*(x))-(m*list[index][0])+(list[index][1])
    return y

# Transforms x, y, z points using M transform matrix
def pTransform(M, x, y, z):
    F = np.reshape(np.array((x, y, z)), (1,3))
    T = (M[0,0]*F[0][0] + M[0,1]*F[0][1] + M[0,2]*F[0][2]), (M[1,0]*F[0][0] + M[1,1]*F[0][1] + M[1,2]*F[0][2]), (M[2,0]*F[0][0] + M[2,1]*F[0][1] + M[2,2]*F[0][2])
    return T

# x parameters for plotting -> xmin, xman, steps
x = np.linspace(-15, 15, 100)

# Center of points
center = (p1[0][0]+p1[1][0]+p1[2][0]+p1[3][0])/4, (p1[0][1]+p1[1][1]+p1[2][1]+p1[3][1])/4

# Test point
test1 = [11, 7]
# Translated around center
test1 = [test1[0]-center[0], test1[1]-center[1]]

# Camera 1
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(figsize=(5, 5))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# Plot lines
plt.plot(x, line(p1, 1, 2, x), color="yellow", linewidth="0.5")
plt.plot(x, line(p1, 2, 3, x), color="white", linewidth="0.5")
plt.plot(x, line(p1, 3, 4, x), color="white", linewidth="0.5")
plt.plot(x, line(p1, 1, 4, x), color="white", linewidth="0.5")
# Plot points
for i in range(len(p1)):
    plt.plot(p1[i][0], p1[i][1], marker="o", markersize=3, markeredgecolor="magenta", markerfacecolor="magenta")
    p1Tran.append(((p1[i][0]-center[0]), (p1[i][1]-center[1])))
    plt.plot(p1Tran[i][0], p1Tran[i][1], marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan")
# Plot center of points
plt.plot(center[0], center[1], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
# Plot transformed center of points/center frame
plt.plot(0, 0, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
# Plot transformed center of points (test2)
plt.plot(test1[0], test1[1], marker=".", markersize=3, markeredgecolor="yellow", markerfacecolor="yellow")

plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.title("Camera View")

# Generates perspective tranform between current points and target points
M1 = cv2.getPerspectiveTransform(np.float32(p1Tran),np.float32(p1Fin))
print(M1)

center2 = (p2[0][0]+p2[1][0]+p2[2][0]+p2[3][0])/4, (p2[0][1]+p2[1][1]+p2[2][1]+p2[3][1])/4

test2 = [1, 8]
test2 = [test2[0]-center2[0], test2[1]-center2[1]]

# Camera 2
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(2, figsize=(5, 5))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# Plot lines
plt.plot(x, line(p2, 1, 2, x), color="yellow", linewidth="0.5")
plt.plot(x, line(p2, 2, 3, x), color="white", linewidth="0.5")
plt.plot(x, line(p2, 3, 4, x), color="white", linewidth="0.5")
plt.plot(x, line(p2, 1, 4, x), color="white", linewidth="0.5")
# Plot points
for i in range(len(p1)):
    plt.plot(p2[i][0], p2[i][1], marker="o", markersize=3, markeredgecolor="magenta", markerfacecolor="magenta")
    p2Tran.append(((p2[i][0]-center2[0]), (p2[i][1]-center2[1])))
    plt.plot(p2Tran[i][0], p2Tran[i][1], marker="o", markersize=3, markeredgecolor="cyan", markerfacecolor="cyan")
# Plot center of points
plt.plot(center2[0], center2[1], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
# Plot transformed center of points/center frame
plt.plot(0, 0, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
# Plot transformed center of points (test2)
plt.plot(test2[0], test2[1], marker=".", markersize=3, markeredgecolor="yellow", markerfacecolor="yellow")

plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.title("Camera 2 View")

# Generates perspective tranform between current points and target points
M2 = cv2.getPerspectiveTransform(np.float32(p2Tran),np.float32(p2Fin))
print(M2)

# Top View
plt.rcParams['axes.facecolor'] = 'black'
plt.figure(3, figsize=(5, 5))
plt.xlim(-15, 15)
plt.ylim(-15, 15)

# Since both transform matricies (M1, M2) should be correct we olny need one set of transformed corners to be shown
for m in range(len(p1Tran)):
    T = pTransform(M1, p1Tran[m][0], p1Tran[m][1], 0)
    pChg.append(T)

# Top View lines drawn
# plt.plot(x, line(pChg, 1, 2, x)) # Isn't function when in top view
plt.plot(x, line(pChg, 2, 3, x), color="white", linewidth="0.5")
# plt.plot(x, line(pChg, 3, 4, x)) # Isn't function when in top view
plt.plot(x, line(pChg, 1, 4, x), color="white", linewidth="0.5")

# On a fully transformed top view there are only vertical and horizontal lines, so need to plot x=? with a vertical line.
plt.axvline(x=-2.5, color="yellow", linewidth="0.5")
plt.axvline(x=2.5, color="white", linewidth="0.5")
for m in range(len(p1Tran)):
    print(pChg[m])
    plt.plot(pChg[m][0], pChg[m][1], marker="o", markersize=4, markeredgecolor="red", markerfacecolor="cyan")

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