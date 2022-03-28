from math import asin, atan2, cos, degrees, pi, sin

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def get_thrust(x):

    # input is numpy array, need to convert to SX
    f = ca.SX(ca.DM(x[13:16]))
    t = ca.SX(ca.DM(x[16:19]))

    # allocation: y = pinv(B) * wrench
    # y = [sin(a1)f1, cos(a1)f1, ... sin(a6)f6, cos(a6)f6]
    # wrench = [f, t]
    # pinv(B) see matlab script allocation.m, with torque = thrust * 0.06
    B_inv = ca.SX([
        [-1/3,            0     ,         0  ,          -25/149,          0     ,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/447,          0     ,       -25/298],
        [ 1/3,            0     ,         0  ,          -25/149,          0     ,       425/894],
        [ 0  ,            0     ,        -1/6,          425/447,          0     ,        25/298],
        [ 1/6,          390/1351,         0  ,           25/298,        695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,          425/894,        760/923 ,       -25/298],
        [-1/6,         -390/1351,         0  ,           25/298,        695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/894,       -760/923 ,        25/298],
        [-1/6,          390/1351,         0  ,           25/298,       -695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/894,        760/923 ,        25/298],
        [ 1/6,         -390/1351,         0  ,           25/298,       -695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,          425/894,       -760/923 ,       -25/298]
    ])
    y = B_inv @ ca.vertcat(f, t)
    f_unit_actuator = ca.vertcat(
        ca.sqrt(y[0]**2 + y[1]**2),
        ca.sqrt(y[2]**2 + y[3]**2),
        ca.sqrt(y[4]**2 + y[5]**2),
        ca.sqrt(y[6]**2 + y[7]**2),
        ca.sqrt(y[8]**2 + y[9]**2),
        ca.sqrt(y[10]**2 + y[11]**2)
    )

    # output is numpy array
    return np.array(ca.DM(f_unit_actuator))

def get_q_w(E):

    roll = E[0]
    pitch = E[1]
    yaw = E[2]
    roll_dot = E[3]
    pitch_dot = E[4]
    yaw_dot = E[5]

    phi = roll
    theta = pitch
    psi = yaw

    cosPhi_2 = cos(phi / 2)
    cosTheta_2 = cos(theta / 2)
    cosPsi_2 = cos(psi / 2)
    sinPhi_2 = sin(phi / 2)
    sinTheta_2 = sin(theta / 2)
    sinPsi_2 = sin(psi / 2)

    q = np.zeros(4)
    q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
    q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
    q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
    q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2

    R = np.array([
        [1,         0,           -sin(theta)],
        [0,  cos(phi), cos(theta) * sin(phi)],
        [0, -sin(phi), cos(theta) * cos(phi)]
    ])
    w = R @ np.array([roll_dot, pitch_dot, yaw_dot])

    return q, w


def q2RPY(q):

    dcm = np.zeros([3, 3])
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    aa = a * a
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d
    dcm[0, 0] = aa + bb - cc - dd
    dcm[0, 1] = 2 * (bc - ad)
    dcm[0, 2] = 2 * (ac + bd)
    dcm[1, 0] = 2 * (bc + ad)
    dcm[1, 1] = aa - bb + cc - dd
    dcm[1, 2] = 2 * (cd - ab)
    dcm[2, 0] = 2 * (bd - ac)
    dcm[2, 1] = 2 * (ab + cd)
    dcm[2, 2] = aa - bb - cc + dd

    theta = asin(-dcm[2, 0])
    if abs(theta - pi / 2) < 1.0e-3:
        phi = 0
        psi = atan2(dcm[1, 2], dcm[0, 2])

    elif abs(theta + pi / 2) < 1.0e-3:
        phi = 0
        psi = atan2(-dcm[1, 2], -dcm[0, 2])

    else:
        phi = atan2(dcm[2, 1], dcm[2, 2])
        psi = atan2(dcm[1, 0], dcm[0, 0])

    # phi = roll, theta = pitch, psi = yaw
    return phi, theta, psi
    


def plot_0(t, p, v, E, w):

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.plot(t, p[0, :])
    plt.xlabel('t')
    plt.ylabel('px')
    plt.subplot(2, 3, 2)
    plt.plot(t, p[1, :])
    plt.xlabel('t')
    plt.ylabel('py')
    plt.subplot(2, 3, 3)
    plt.plot(t, p[2, :])
    plt.xlabel('t')
    plt.ylabel('pz')
    plt.subplot(2, 3, 4)
    plt.plot(t, v[0, :])
    plt.xlabel('t')
    plt.ylabel('vx')
    plt.subplot(2, 3, 5)
    plt.plot(t, v[1, :])
    plt.xlabel('t')
    plt.ylabel('vy')
    plt.subplot(2, 3, 6)
    plt.plot(t, v[2, :])
    plt.xlabel('t')
    plt.ylabel('vz')

    plt.figure(2)
    plt.subplot(3, 3, 1)
    plt.plot(t, E[0, :])
    plt.xlabel('t')
    plt.ylabel('R')
    plt.subplot(3, 3, 2)
    plt.plot(t, E[1, :])
    plt.xlabel('t')
    plt.ylabel('P')
    plt.subplot(3, 3, 3)
    plt.plot(t, E[2, :])
    plt.xlabel('t')
    plt.ylabel('Y')
    plt.subplot(3, 3, 4)
    plt.plot(t, E[3, :])
    plt.xlabel('t')
    plt.ylabel('dR')
    plt.subplot(3, 3, 5)
    plt.plot(t, E[4, :])
    plt.xlabel('t')
    plt.ylabel('dP')
    plt.subplot(3, 3, 6)
    plt.plot(t, E[5, :])
    plt.xlabel('t')
    plt.ylabel('dY')
    plt.subplot(3, 3, 7)
    plt.plot(t, w[0, :])
    plt.xlabel('t')
    plt.ylabel('wx')
    plt.subplot(3, 3, 8)
    plt.plot(t, w[1, :])
    plt.xlabel('t')
    plt.ylabel('wy')
    plt.subplot(3, 3, 9)
    plt.plot(t, w[2, :])
    plt.xlabel('t')
    plt.ylabel('wz')

    plt.show()

def compare_plot(trajectory, state, u, solver_time):
    t = trajectory[0, :]
    p = trajectory[1:4, :]
    q = trajectory[4:8, :]
    v = trajectory[8:11, :]
    w = trajectory[11:14, :]
    E = np.zeros_like(p)

    xp = state[0:3, :]
    xq = state[3:7, :]
    xv = state[7:10, :]
    xw = state[10:13, :]
    xE = np.zeros_like(E)
    xf = state[13:16, :]
    xt = state[16:19, :]
    Fa = np.zeros([6, len(t)])

    for i in range(len(t)):
        R, P, Y = q2RPY(q[:, i])
        E[0, i] = R
        E[1, i] = P
        E[2, i] = Y

        xR, xP, xY = q2RPY(xq[:, i])
        xE[0, i] = xR
        xE[1, i] = xP
        xE[2, i] = xY

        Fa[:, i] = get_thrust(state[:, i])[:, 0]

    plt.figure('translation')
    plt.subplot(2, 3, 1)
    plt.plot(t, p[0, :])
    plt.plot(t, xp[0, :])
    plt.xlabel('t')
    plt.ylabel('px')
    plt.subplot(2, 3, 2)
    plt.plot(t, p[1, :])
    plt.plot(t, xp[1, :])
    plt.xlabel('t')
    plt.ylabel('py')
    plt.subplot(2, 3, 3)
    plt.plot(t, p[2, :])
    plt.plot(t, xp[2, :])
    plt.xlabel('t')
    plt.ylabel('pz')
    plt.subplot(2, 3, 4)
    plt.plot(t, v[0, :])
    plt.plot(t, xv[0, :])
    plt.xlabel('t')
    plt.ylabel('vx')
    plt.subplot(2, 3, 5)
    plt.plot(t, v[1, :])
    plt.plot(t, xv[1, :])
    plt.xlabel('t')
    plt.ylabel('vy')
    plt.subplot(2, 3, 6)
    plt.plot(t, v[2, :])
    plt.plot(t, xv[2, :])
    plt.xlabel('t')
    plt.ylabel('vz')
    
    plt.figure('rotation')
    plt.subplot(2, 3, 1)
    plt.plot(t, E[0, :])
    plt.plot(t, xE[0, :])
    plt.xlabel('t')
    plt.ylabel('R')
    plt.subplot(2, 3, 2)
    plt.plot(t, E[1, :])
    plt.plot(t, xE[1, :])
    plt.xlabel('t')
    plt.ylabel('P')
    plt.subplot(2, 3, 3)
    plt.plot(t, E[2, :])
    plt.plot(t, xE[2, :])
    plt.xlabel('t')
    plt.ylabel('Y')
    plt.subplot(2, 3, 4)
    plt.plot(t, w[0, :])
    plt.plot(t, xw[0, :])
    plt.xlabel('t')
    plt.ylabel('wx')
    plt.subplot(2, 3, 5)
    plt.plot(t, w[1, :])
    plt.plot(t, xw[1, :])
    plt.xlabel('t')
    plt.ylabel('wy')
    plt.subplot(2, 3, 6)
    plt.plot(t, w[2, :])
    plt.plot(t, xw[2, :])
    plt.xlabel('t')
    plt.ylabel('wz')

    plt.figure('xyzRPY')
    plt.subplot(3, 2, 1)
    plt.plot(t, p[0, :])
    plt.plot(t, xp[0, :])
    plt.xlabel('t')
    plt.ylabel('px')
    plt.subplot(3, 2, 2)
    plt.plot(t, degrees(1) * E[0, :])
    plt.plot(t, degrees(1) * xE[0, :])
    plt.xlabel('t')
    plt.ylabel('R')
    plt.subplot(3, 2, 3)
    plt.plot(t, p[1, :])
    plt.plot(t, xp[1, :])
    plt.xlabel('t')
    plt.ylabel('py')
    plt.subplot(3, 2, 4)
    plt.plot(t, degrees(1) * E[1, :])
    plt.plot(t, degrees(1) * xE[1, :])
    plt.xlabel('t')
    plt.ylabel('P')
    plt.subplot(3, 2, 5)
    plt.plot(t, p[2, :])
    plt.plot(t, xp[2, :])
    plt.xlabel('t')
    plt.ylabel('pz')
    plt.subplot(3, 2, 6)
    plt.plot(t, E[2, :])
    plt.plot(t, xE[2, :])
    plt.xlabel('t')
    plt.ylabel('Y')

    plt.figure('thrust')
    plt.plot(t, Fa[0, :])
    plt.plot(t, Fa[1, :])
    plt.plot(t, Fa[2, :])
    plt.plot(t, Fa[3, :])
    plt.plot(t, Fa[4, :])
    plt.plot(t, Fa[5, :])
    plt.xlabel('t')
    plt.ylabel('force')

    plt.figure('wrench')
    plt.subplot(6, 2, 1)
    plt.plot(t, xf[0, :])
    plt.xlabel('t')
    plt.ylabel('fx')
    plt.subplot(6, 2, 3)
    plt.plot(t, xf[1, :])
    plt.xlabel('t')
    plt.ylabel('fy')
    plt.subplot(6, 2, 5)
    plt.plot(t, xf[2, :])
    plt.xlabel('t')
    plt.ylabel('fz')
    plt.subplot(6, 2, 7)
    plt.plot(t, xt[0, :])
    plt.xlabel('t')
    plt.ylabel('tx')
    plt.subplot(6, 2, 9)
    plt.plot(t, xt[1, :])
    plt.xlabel('t')
    plt.ylabel('ty')
    plt.subplot(6, 2, 11)
    plt.plot(t, xt[2, :])
    plt.xlabel('t')
    plt.ylabel('tz')
    plt.subplot(6, 2, 2)
    plt.plot(t, u[0, :])
    plt.xlabel('t')
    plt.ylabel('fx_dot')
    plt.subplot(6, 2, 4)
    plt.plot(t, u[1, :])
    plt.xlabel('t')
    plt.ylabel('fy_dot')
    plt.subplot(6, 2, 6)
    plt.plot(t, u[2, :])
    plt.xlabel('t')
    plt.ylabel('fz_dot')
    plt.subplot(6, 2, 8)
    plt.plot(t, u[3, :])
    plt.xlabel('t')
    plt.ylabel('tx_dot')
    plt.subplot(6, 2, 10)
    plt.plot(t, u[4, :])
    plt.xlabel('t')
    plt.ylabel('ty_dot')
    plt.subplot(6, 2, 12)
    plt.plot(t, u[5, :])
    plt.xlabel('t')
    plt.ylabel('tz_dot')

    plt.figure('solver_time')
    plt.plot(t, solver_time)

    plt.show()


def generate_trajectory_0(T, t_1, t_2, r, h, d_h, dt, do_print=False, do_plot=False):

    timestamp = np.arange(0.0, T + dt, dt)

    length = len(timestamp)

    p = np.zeros([3, length])  # analytic, xyz
    v = np.zeros([3, length])  # analytic, xyz, time derivative by hand
    E = np.zeros([6, length])  # analytic, ZYX Euler angle, RPY, RPY_dot

    w = np.zeros([3, length])  # converted from E
    q = np.zeros([4, length])  # converted from E

    # calculate trajectory
    for i in range(length):
        
        t = timestamp[i]

        # piecewise continuous trajectory
        if t >= 0 and t < t_1:

            p[0, i] = 0
            p[1, i] = r / t_1 * t
            p[2, i] = -h - d_h + d_h * cos(2 * pi / (2 * t_1 + t_2) * t)

            v[0, i] = 0
            v[1, i] = r / t_1
            v[2, i] = -d_h * 2 * pi / (2 * t_1 + t_2) * sin(2 * pi / (2 * t_1 + t_2) * t)
            
            E[0, i] = 0
            E[1, i] = -v[2, i]
            E[2, i] = pi / 2
            E[3, i] = 0
            E[4, i] = 4 * pi**2 * d_h * cos(2 * pi * t / (2 * t_1 + t_2)) / (2 * t_1 + t_2)**2
            E[5, i] = 0

            q[:, i], w[:, i] = get_q_w(E[:, i])

        if t >= t_1 and t < t_1 + t_2:
            
            omega = 3 * pi / (2 * t_2)
            
            p[0, i] = -r + r * cos(omega * (t - t_1))
            p[1, i] = r + r * sin(omega * (t - t_1))
            p[2, i] = -h - d_h + d_h * cos(2 * pi / (2 * t_1 + t_2) * t)

            v[0, i] = -r * omega * sin(omega * (t - t_1))
            v[1, i] = r * omega * cos(omega * (t - t_1))
            v[2, i] = -d_h * 2 * pi / (2 * t_1 + t_2) * sin(2 * pi / (2 * t_1 + t_2) * t)
            
            E[0, i] = pi / 4 - pi / 4 * cos(2 * pi / t_2 * (t - t_1))
            E[1, i] = -v[2, i]
            E[2, i] = pi / 2 + omega * (t - t_1)
            E[3, i] = pi / 4 * 2 * pi / t_2 * sin(2 * pi / t_2 * (t - t_1))
            E[4, i] = 4 * pi**2 * d_h * cos(2 * pi * t / (2 * t_1 + t_2)) / (2 * t_1 + t_2)**2
            E[5, i] = omega

            q[:, i], w[:, i] = get_q_w(E[:, i])

        if t >= t_1 + t_2 and t < 3 * t_1 + t_2:
            
            p[0, i] = -r + r / t_1 * (t - t_1 - t_2)
            p[1, i] = 0
            p[2, i] = -h - d_h + d_h * cos(2 * pi / (2 * t_1 + t_2) * t)

            v[0, i] = r / t_1
            v[1, i] = 0
            v[2, i] = -d_h * 2 * pi / (2 * t_1 + t_2) * sin(2 * pi / (2 * t_1 + t_2) * t)
            
            E[0, i] = 0
            E[1, i] = -v[2, i]
            E[2, i] = 0
            E[3, i] = 0
            E[4, i] = 4 * pi**2 * d_h * cos(2 * pi * t / (2 * t_1 + t_2)) / (2 * t_1 + t_2)**2
            E[5, i] = 0

            q[:, i], w[:, i] = get_q_w(E[:, i])

        if t >= 3 * t_1 + t_2 and t < 3 * T - t_1:

            omega = 3 * pi / (2 * t_2)

            p[0, i] = r + r * sin(omega * (t - 3 * t_1 - t_2))
            p[1, i] = -r + r * cos(omega * (t - 3 * t_1 - t_2))
            p[2, i] = -h - d_h + d_h * cos(2 * pi / (2 * t_1 + t_2) * t)

            v[0, i] = r * omega * cos(omega * (t - 3 * t_1 - t_2))
            v[1, i] = -r * omega * sin(omega * (t - 3 * t_1 - t_2))
            v[2, i] = -d_h * 2 * pi / (2 * t_1 + t_2) * sin(2 * pi / (2 * t_1 + t_2) * t)
            
            E[0, i] = -pi / 4 + pi / 4 * cos(2 * pi / t_2 * (t - 3 * t_1 - t_2))
            E[1, i] = -v[2, i]
            E[2, i] = -omega * (t - 3 * t_1 - t_2)
            E[3, i] = -pi / 4 * 2 * pi / t_2 * sin(2 * pi / t_2 * (t - 3 * t_1 - t_2))
            E[4, i] = 4 * pi**2 * d_h * cos(2 * pi * t / (2 * t_1 + t_2)) / (2 * t_1 + t_2)**2
            E[5, i] = -omega

            q[:, i], w[:, i] = get_q_w(E[:, i])

        if t >= T - t_1 and t <= T:

            p[0, i] = 0
            p[1, i] = -r + r / t_1 * (t - (T - t_1))
            p[2, i] = -h - d_h + d_h * cos(2 * pi / (2 * t_1 + t_2) * t)

            v[0, i] = 0
            v[1, i] = r / t_1
            v[2, i] = -d_h * 2 * pi / (2 * t_1 + t_2) * sin(2 * pi / (2 * t_1 + t_2) * t)
            
            E[0, i] = 0
            E[1, i] = -v[2, i]
            E[2, i] = pi / 2
            E[3, i] = 0
            E[4, i] = 4 * pi**2 * d_h * cos(2 * pi * t / (2 * t_1 + t_2)) / (2 * t_1 + t_2)**2
            E[5, i] = 0

            q[:, i], w[:, i] = get_q_w(E[:, i])

        # limit yaw
        while E[2, i] > pi:
            E[2, i] -= 2 * pi
        while E[2, i] < -pi:
            E[2, i] += 2 * pi

        # print
        if do_print:

            print('--------------------------------------------------')
            print(i)
            print(timestamp[i])
            print(p[:, i])
            print(q[:, i])
            print(v[:, i])
            print(w[:, i])
            print(E[:, i])

    # visualization
    if do_plot:
        plot_0(timestamp, p, v, E, w)
    
    # return variables needed by MPC
    trajectory = np.vstack((timestamp, p, q, v, w))

    return trajectory


if __name__ == '__main__':

    T = 40
    t_1 = T / (3 * pi + 4)
    t_2 = 3 * pi * T / (6 * pi + 8)
    r = 1.5
    h = 2.5
    d_h = 0.5
    dt = 0.01  # controller runs at 100 hz

    trajectory = generate_trajectory_0(T, t_1, t_2, r, h, d_h, dt, do_print=False, do_plot=True)

    x = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0, 0,0,-9.8*6, 1,0,0])
    rt = get_thrust(x)
    print(rt[:, 0])
