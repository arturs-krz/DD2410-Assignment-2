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

# returns the DH table filled with values for a given joint configuration
def get_kuka_DH(q):
    kuka_DH = [
        DH_Entry(math.pi / 2., 0.311, 0., q[0]), # add base offset as d
        DH_Entry(-math.pi / 2., 0., 0., q[1]),
        DH_Entry(-math.pi / 2., 0.4, 0., q[2]),
        DH_Entry(math.pi / 2., 0., 0., q[3]),
        DH_Entry(math.pi / 2., 0.39, 0., q[4]),
        DH_Entry(-math.pi / 2., 0., 0., q[5]),
        DH_Entry(0., 0.078, 0., q[6]), # add end effector offset as d
    ]
    return kuka_DH

# Calculates the full transform for i to i-1
# Takes the DH parameters
def transform_to_prev(theta, d, alpha, r):
    transform = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1.]
    ])
    return transform

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements
    R = np.array(R)
    """
    Fill in your IK solution here and return the seven joint values in q
    """

    # eps_theta = 0.1 * np.ones(len(joint_positions))
    theta_hat = np.array(joint_positions)

    # angle_X, angle_Y, angle_Z = get_euler_angles_from_T(np.array(R))
    X = np.array([x, y, z])

    X_hat, R_hat, intermediary_transforms = kuka_FK(theta_hat)
    eps_X = X_hat - X
    # calculate the error on rotation from the rotation matrices directly, euler angles are buggy
    eps_R = (np.cross(R[:,0], R_hat[:,0]) + np.cross(R[:,1], R_hat[:,1]) + np.cross(R[:,2], R_hat[:,2])) / 2.

    eps = np.concatenate([eps_X, eps_R])
    tolerance = 0.001

    # iterate the approximation until we get under the min tolerance
    
    MAX_ITERATIONS = 300
    iterations = 0
    while np.mean(np.abs(eps)) >= tolerance and iterations < MAX_ITERATIONS:
        J, J_inv = compute_jacobian(intermediary_transforms, theta_hat)
        eps_theta = np.dot(J_inv, eps)
        # if iterations % 50 == 0:
        #     print(eps_theta)
        # print(eps_theta)
        theta_hat = theta_hat - eps_theta
        
        X_hat, R_hat, intermediary_transforms = kuka_FK(theta_hat)
        eps_X = X_hat - X
        eps_R = (np.cross(R[:,0], R_hat[:,0]) + np.cross(R[:,1], R_hat[:,1]) + np.cross(R[:,2], R_hat[:,2])) / 2.
        eps = np.concatenate([eps_X, eps_R])

        iterations = iterations + 1
        # print(iterations)
        # print(eps_X)

    # print(X_hat)
    q = theta_hat.tolist()
    return q

# BUGGY
def get_euler_angles_from_T(T):
    # Euler angles from the rotation part of the matrix
    #     | r11 r12 r13 |
    # R = | r21 r22 r23 |
    #     | r31 r32 r33 |

    # Method as per https://walter.readthedocs.io/en/latest/Kinematics/
    # and http://www.gregslabaugh.net/publications/euler.pdf
    alpha = 0.
    beta = 0.
    gamma = 0.

    if np.abs(T[2,0] + 1.) < 0.03:
        # print("NEGATIVE")
        beta = math.pi / 2.
        alpha = math.atan2(T[0,1], T[0,2])
        gamma = 0.
    elif np.abs(T[2,0] - 1.) < 0.03:
        # print("POSITIVE")
        beta = -math.pi / 2.
        alpha = math.atan2(-T[0,1], -T[0,2])
        gamma = 0.
    else:
        # print("BASECASE")
        beta = -math.asin(T[2,0])
        cosBeta = math.cos(beta)
        alpha = math.atan2(T[2,1] / cosBeta, T[2,2] / cosBeta)
        gamma = math.atan2(T[1,0] / cosBeta, T[0,0] / cosBeta)

    # print(alpha, beta, gamma)
    return alpha, beta, gamma


# Forward kinematics
def kuka_FK(q):

    # end effector transform in the current frame
    current_T = np.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    # get the DH table values for the current q values
    DH_values = get_kuka_DH(q)

    # iterate through the joints
    # save intermediary transforms needed for Jacobian
    intermediaries = [current_T]
    for i in range(len(q)):
        link = DH_values[i] # get corresponding DH row
        transform = transform_to_prev(link.theta, link.d, link.alpha, link.r)

        current_T = np.matmul(current_T, transform)
        intermediaries.append(current_T)

    # angle_X, angle_Y, angle_Z = get_euler_angles_from_T(current_T)
    
    # split the position and rotation components, can't handle euler angles correctly :/
    X = current_T[:3,3]
    R = current_T[:3,:3]
    
    return X, R, intermediaries

def compute_jacobian(transforms, q):
    #     | Jp (3 x n) |
    # J = | Jo ( 3 x n) |

    # p are position vectors for the origin frames (link frames)
    # p_e for the end effector frame
    p_e = transforms[len(transforms) - 1][0:3,3] # get the (x,y,z) vector for the last transform

    num_cols = len(q)
    J = np.zeros((6, num_cols))
    for i in range(num_cols):
        # all joints revolute, so no need to cover prismatic case
        
        # linear:
        # z_i-1 X (p_e - p_i-1)
        p = transforms[i][0:3,3] # (x,y,z)

        # z is unit vector of joint axis (z axis)
        # so we need the rotation of z component
        z = transforms[i][0:3,2]
        z = z / np.sqrt(np.sum(np.square(z))) # normalize to get unit vec

        linear = np.cross(z, p_e - p)

        # angular is just z_i-1
        column = np.array([linear[0], linear[1], linear[2], z[0], z[1], z[2]])
        J[:,i] = column

    return J, np.linalg.pinv(J)