#! /usr/bin/env python3
import math
import numpy as np
"""
    # Arturs Kurzemnieks
    # artursk@kth.se
"""

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]

    """
    Fill in your IK solution here and return the three joint values in q
    """

    L0 = 0.07
    x = x - L0  # compensate the offset of the first joint

    L1 = 0.3
    L2 = 0.35

    L1square = L1 * L1
    L2square = L2 * L2


    Csquare = (x*x) + (y*y) # square of the (x,y) vector as expressed with x and y using pythagorean
    C = math.sqrt(Csquare)
    # it can also be expressed using L1 and L2 using the law of cosines (generalization)
    # Csquare == L1square + L2square - 2*L1*L2*cos(alpha), where alpha = PI - q2,
    # since cos(PI - theta) = -cos(theta), we can drop the minus sign from division and get cosine for q2 directly

    cosq2 = (Csquare - L1square - L2square) / (2 * L1 * L2)

    # cos^2theta + sin^2theta = 1
    sinq2 = math.sqrt(1 - (cosq2*cosq2))

    # sinq2 / cosq2 = tangentq2 = opposite / adjacent edge
    # so we can use arctan2 to get q2 angle
    q2 = math.atan2(sinq2, cosq2)


    # construct a "virtual" right triangle between the origin, (x,y) and a new point on an extended L1
    x1 = L1 + (L2 * cosq2) # the"extended" L1 that starts at origin and ends at the new point  
    y1 = L2 * sinq2  # the other edge between the new point and (x,y)

    # total angle between vector (x,y) and x axis - the angle of the new triangle near the origin
    q1 = math.atan2(y,x) - math.atan2(y1, x1)

    q = [q1, q2, z]

    return q

class DH_Entry:
    def __init__(self, alpha, d, r, theta):
        self.alpha = alpha
        self.d = d
        self.r = r
        self.theta = theta

def get_kuka_DH(q):
    kuka_DH = [
        DH_Entry(math.pi / 2, 0, 0, q[0]),
        DH_Entry(-math.pi / 2, 0, 0, q[1]),
        DH_Entry(-math.pi / 2, 0.4, 0, q[2]),
        DH_Entry(math.pi / 2, 0, 0, q[3]),
        DH_Entry(math.pi / 2, 0.39, 0, q[4]),
        DH_Entry(-math.pi / 2, 0, 0, q[5]),
        DH_Entry(0, 0, 0, q[6])
    ]
    return kuka_DH

# Calculates the full transform for i to i-1
# Takes the DH parameters
def transform_to_prev(theta, d, alpha, r):
    transform = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return transform

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """
    # forward kinematics for q

    # eps_theta = 0.01 * np.ones(7)
    # theta_hat = np.array(q) + eps_theta
    # print theta_hat

    # X = np.array([x, y, z, euler values from R])
    # X_hat = kuka_FK(theta_hat)
    # eps_X = X_hat - X

    tolerance = 1e-5
    # iterate the approximation until we get under the min tolerance
    # while eps_X > tolerance
    print(kuka_FK(joint_positions))
    

    return q

# Forward kinematics
def kuka_FK(q):

    # end effector transform in the current frame
    # initially translated 0.311 in z direction due to the base offset
    current_T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.311],
        [0, 0, 0, 1]
    ])

    # get the DH table values for the current q values
    DH_values = get_kuka_DH(q)

    # iterate through the joints
    for i in range(len(q)):
        link = DH_values[i] # get corresponding DH row
        transform = transform_to_prev(link.theta, link.d, link.alpha, link.r)

        current_T = np.dot(current_T, transform)
    
    # + end effector offset
    # current_T = 

    # Euler angles from the rotation part of the matrix
    #     | r11 r12 r13 |
    # R = | r21 r22 r23 |
    #     | r31 r32 r33 |

    angle_X = np.arctan2(current_T[2,1], current_T[2,2])
    angle_Y = np.arctan2(-current_T[2,0], np.sqrt(np.square(current_T[2,1]) + np.square(current_T[2,2])))
    angle_Z = np.arctan2(current_T[1,0], current_T[0,0])

    return [current_T[0,3], current_T[1,3], current_T[2,3], angle_X, angle_Y, angle_Z]

