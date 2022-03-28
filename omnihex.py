'''

Created by Yueqian Liu (yueqianliu@outlook.com)

examples are copied from the following repos
diff_drive_robot https://github.com/tomcattiger1230/ACADOS_Example
data_driven_mpc https://github.com/uzh-rpg/data_driven_mpc

'''

import math
import shutil
import timeit
from os import environ, path, removedirs, system

import matplotlib.pyplot as plt
import numpy as np
from acados_template import (AcadosModel, AcadosOcp, AcadosOcpSolver,
                             AcadosSim, AcadosSimSolver)
from casadi import (DM, SX, Function, cross, diag, dot, inv, norm_2, sqrt,
                    vertcat)

from sx_quaternion import (quat_derivative, quat_err, rotate_vec3,
                           sx_quat_inverse, sx_quat_multiply,
                           sx_quat_normalize)
from trajectory import compare_plot, generate_trajectory_0


def get_omnihex_ode_model():

    # nomial value of parameters
    m = 4.0
    g = SX([0, 0, 9.8])
    com = SX([0, 0, 0.03])  # center of mass position in body FRD frame
    J_xx = 0.08401764152
    J_xy = -0.00000445135
    J_xz = 0.00014163105
    J_yy = 0.08169689992
    J_yz = -0.00000936372
    J_zz = 0.14273598018
    J = SX([
        [J_xx, J_xy, J_xz],
        [J_xy, J_yy, J_yz],
        [J_xz, J_yz, J_zz]
    ])
    J_inv = inv(J)

    # states and controls
    p = SX.sym('p', 3)  # position in world NED frame (m)
    q = SX.sym('q', 4)  # attitude expressed in quaternion
    v = SX.sym('v', 3)  # velocity in world frame (m/s)
    w = SX.sym('w', 3)  # angular velocity in body FRD frame (rad/s)
    f = SX.sym('f', 3)  # actuator force in body frame (N)
    t = SX.sym('t', 3)  # actuator torque in body frame (Nm)

    p_dot = SX.sym('p_dot', 3)
    q_dot = SX.sym('q_dot', 4)
    v_dot = SX.sym('v_dot', 3)
    w_dot = SX.sym('w_dot', 3)
    f_dot_x = SX.sym('f_dot_x', 3)
    t_dot_x = SX.sym('t_dot_x', 3)
    f_dot_u = SX.sym('f_dot_u', 3)
    t_dot_u = SX.sym('t_dot_u', 3)

    x = vertcat(p, q, v, w, f, t)
    u = vertcat(f_dot_u, t_dot_u)

    # reference trajectory, or state reference
    # using the parameter field as the interface
    p_ref = SX.sym('p_ref', 3)
    q_ref = SX.sym('q_ref', 4)
    v_ref = SX.sym('v_ref', 3)
    w_ref = SX.sym('w_ref', 3)
    p = vertcat(p_ref, q_ref, v_ref, w_ref)

    # dynamics
    x_dot = vertcat(p_dot, q_dot, v_dot, w_dot, f_dot_x, t_dot_x)
    f_expl = vertcat(
        v,
        quat_derivative(q, w),
        rotate_vec3(f, q) / m + g,
        J_inv @ (t + cross(com, f) - cross(w, J @ w)),
        f_dot_u,
        t_dot_u
    )

    f_impl = x_dot - f_expl

    # acados model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.p = p
    model.name = 'omnihex'
    return model


def e(x, ref):

    p_ref = ref[0:3]
    q_ref = ref[3:7]
    v_ref = ref[7:10]
    w_ref = ref[10:13]

    p_err = x[0:3] - p_ref
    q_err = quat_err(x[3:7], q_ref)
    v_err = x[7:10] - v_ref
    w_err = x[10:13] - w_ref
    f = x[13:16]
    t = x[16:19]

    return vertcat(p_err, q_err, v_err, w_err, f, t)


def h(x):

    # fields in the state variable
    v = x[7:10]
    w = x[10:13]
    f = x[13:16]
    t = x[16:19]

    # allocation: y = pinv(B) * wrench
    # y = [sin(a1)f1, cos(a1)f1, ... sin(a6)f6, cos(a6)f6]
    # wrench = [f, t]
    # pinv(B) see matlab script allocation.m, with torque = thrust * 0.06
    B_inv = SX([
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
    y = B_inv @ vertcat(f, t)
    f_unit_actuator = vertcat(
        sqrt(y[0]**2 + y[1]**2),
        sqrt(y[2]**2 + y[3]**2),
        sqrt(y[4]**2 + y[5]**2),
        sqrt(y[6]**2 + y[7]**2),
        sqrt(y[8]**2 + y[9]**2),
        sqrt(y[10]**2 + y[11]**2)
    )

    return vertcat(v, w)#, f_unit_actuator)



def formulate_omnihex_ocp():

    # cost weighting matrices
    Q_p = SX([1, 1, 1]) * 10
    Q_q = SX([2, 4, 1]) * 15
    Q_v = SX([1, 1, 1]) * 5
    Q_w = SX([1, 1, 1]) * 5
    Q_f = SX([1, 1, 1]) * 0.001
    Q_t = SX([1, 1, 1]) * 0.001
    Q = diag(vertcat(Q_p, Q_q, Q_v, Q_w, Q_f, Q_t))

    # small R -> aggressive moves, large R -> large error
    R_f_dot = SX([1, 1, 1]) * 0.001
    R_t_dot = SX([1, 1, 1]) * 0.001
    R = diag(vertcat(R_f_dot, R_t_dot))

    Q_p_e = SX([1, 1, 1]) * 10
    Q_q_e = SX([2, 4, 1]) * 15
    Q_v_e = SX([1, 1, 1]) * 5
    Q_w_e = SX([1, 1, 1]) * 5
    Q_f_e = SX([1, 1, 1]) * 0.001
    Q_t_e = SX([1, 1, 1]) * 0.001
    Q_e = diag(vertcat(Q_p_e, Q_q_e, Q_v_e, Q_w_e, Q_f_e, Q_t_e))

    # constraints, u = [f_dot, t_dot], h = [v, w, f_unit_actuator]
    u_lb = np.array([-1, -1, -1, -1, -1, -1]) * 100
    u_ub = np.array([ 1,  1,  1,  1,  1,  1]) * 100
    h_lb = np.array([-10, -10, -10, -90, -90, -90]) #, 0, 0, 0, 0, 0, 0])
    h_ub = np.array([10, 10, 10, 90, 90, 90])#, 20, 20, 20, 20, 20, 20])

    # optimal control problem
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp
    ocp = AcadosOcp()

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpDims
    # prediction horizon only, other fields are set automatically
    ocp.dims.N = 20

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp.model
    ocp.model = get_omnihex_ode_model()

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    ocp.model.con_h_expr = h(x)
    ocp.model.con_h_expr_e = h(x)
    ocp.model.cost_expr_ext_cost = e(x, p).T @ Q @ e(x, p) + u.T @ R @ u
    ocp.model.cost_expr_ext_cost_e = e(x, p).T @ Q_e @ e(x, p)

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpConstraints
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = u_lb
    ocp.constraints.ubu = u_ub
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.lh = h_lb
    ocp.constraints.uh = h_ub
    ocp.constraints.lh_e = h_lb
    ocp.constraints.uh_e = h_ub
    ocp.constraints.x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpOptions
    ocp.solver_options.tf = 1.0
    ocp.solver_options.qp_solver_iter_max = 1000
    ocp.solver_options.levenberg_marquardt = 1e-3
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'

    ocp.acados_include_path = environ['ACADOS_SOURCE_DIR'] + '/include'
    ocp.acados_lib_path = environ['ACADOS_SOURCE_DIR'] + '/lib'
    ocp.code_export_directory = './acados_export'

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp.parameter_values
    ocp.parameter_values = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return ocp


def get_sim_integrator(ocp, dt):

    # https://docs.acados.org/python_interface/index.html#acados-integrator-interface
    sim = AcadosSim()

    sim.model = ocp.model
    sim.acados_include_path = ocp.acados_include_path
    sim.acados_lib_path = ocp.acados_lib_path
    sim.parameter_values = ocp.parameter_values
    sim.code_export_directory = ocp.code_export_directory

    sim.solver_options.T = dt
    sim.solver_options.newton_iter = 3
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1

    return sim


if __name__ == '__main__':

    np.set_printoptions(linewidth=1e6, suppress=True, precision=4)

    # ocp solver
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_solver.AcadosOcpSolver
    ocp = formulate_omnihex_ocp()

    if path.exists(ocp.code_export_directory):
        shutil.rmtree(ocp.code_export_directory)

    ocp_json_path = ocp.code_export_directory + '/acados_ocp_' + ocp.model.name + '.json'
    ocpSolver = AcadosOcpSolver(ocp, ocp_json_path)

    # simulation integrator
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_sim_solver.AcadosSimSolver
    control_rate = 100
    dt = 1 / control_rate
    sim = get_sim_integrator(ocp, dt)

    sim_json_path = sim.code_export_directory + '/acados_sim_' + sim.model.name + '.json'
    simSolver = AcadosSimSolver(sim, sim_json_path)

    # get trajectory
    T = 10
    t_1 = T / (3 * math.pi + 4)
    t_2 = 3 * math.pi * T / (6 * math.pi + 8)
    r = 1.5
    H = 2.5
    d_h = 0.5
    trajectory = generate_trajectory_0(T, t_1, t_2, r, H, d_h, dt, do_print=False, do_plot=False)

    dimension = np.size(trajectory, 0)
    length = np.size(trajectory, 1)
    print('total trajectory size: [%d, %d]' %(dimension, length))

    # test mpc
    x = np.zeros([ocp.dims.nx, length])
    x[0:7, 0] = trajectory[1:8, 0]
    x[15, 0] = -9.8 * 4

    u = np.zeros([ocp.dims.nu, length])

    solver_time = np.zeros(length)

    for i in range(length):

        # current time
        t = trajectory[0, i]
        print('t =', t)

        # get reference within prediction horizon
        row = dimension - 1
        col = ocp.dims.N + 1
        p = np.zeros([row, col])
        
        for j in range(col):

            idx = i + j * int(ocp.solver_options.Tsim / sim.solver_options.T)

            if idx >= length:
                p[:, j] = trajectory[1:, length - 1]
                p[9:, j] = np.zeros_like(p[9:, j])

            else:
                p[:, j] = trajectory[1:, idx]

            ocpSolver.set(j, 'p', p[:, j])

        # set solver initial state, must set initial constraints
        ocpSolver.set(0, 'x', x[:, i])
        ocpSolver.constraints_set(0, 'lbx', x[:, i])
        ocpSolver.constraints_set(0, 'ubx', x[:, i])

        # solve for f_dot and t_dot
        start = timeit.default_timer()
        ocp_status = ocpSolver.solve()
        ocpSolver.print_statistics()
        stop = timeit.default_timer()
        solver_time[i] = stop - start
        print('solver_time =', solver_time[i])

        if ocp_status != 0:
            print('OCP_SOL: ocp_status =', ocp_status, ', exiting')
            quit()

        u[:, i] = ocpSolver.get(0, 'u')
        print(u[:, i])

        # simulate dynamics and update x using acados sim
        simSolver.set('x', x[:, i])
        simSolver.set('u', u[:, i])

        sim_status = simSolver.solve()

        if sim_status != 0:
            print('SIM_SOL: sim_status =', sim_status, ', exiting')
            quit()

        if i + 1 < length:
            x[:, i + 1] = simSolver.get('x')

    compare_plot(trajectory, x, u, solver_time)
