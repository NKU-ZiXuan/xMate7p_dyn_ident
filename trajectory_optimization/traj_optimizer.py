from pyOpt import pySLSQP
from pyOpt import pyNSGA2
from pyOpt import pyNLPQL
from pyOpt import pySOLVOPT
import pyOpt
import numpy as np
import matplotlib.pyplot as plt
from fourier_traj import FourierTraj
import csv
import sympy

import sympybotics as spb

q0_scale = np.pi
fourier_scale = 10*np.pi

# joint constraints
# [(joint_var, q_low, q_upper, dq_low, dq_upper), ..., (...)]

# cartesian constraints
# [(joint_num, x_low, x_high, y_low, y_high, z_low, z_high), ..., (...)]

class TrajOptimizer:
    def __init__(self, dyn, order, base_freq, joint_constraints=[], cartesian_constraints=[],
                 q0_min=-q0_scale, q0_max=q0_scale,
                 ab_min=-fourier_scale, ab_max=fourier_scale, verbose=False):
        self._order = order
        self._base_freq = base_freq
        self._dyn = dyn
        self._joint_constraints = joint_constraints
        self._dof_num = len(self._joint_constraints[0])
        print('dof_num: {}'.format(self._dof_num))
        self._joint_const_num = len(self._joint_constraints[0]) * (len(self._joint_constraints[1])-1)
        print('joint constraint number: {}'.format(self._joint_const_num))
        self._cartesian_constraints = cartesian_constraints
        self._cartesian_const_num = len(self._cartesian_constraints)
        print('cartesian constraint number: {}'.format(self._cartesian_const_num))
        self._const_num = self._joint_const_num + self._cartesian_const_num * 3
        print('constraint number: {}'.format(self._const_num))

        self._q0_min = q0_min
        self._q0_max = q0_max
        self._ab_min = ab_min
        self._ab_max = ab_max

        # sample number for the highest term
        self._sample_point = 200

        self.fourier_traj = FourierTraj(self._dyn.dof, self._order, self._base_freq,
                                        sample_num_per_period=self._sample_point)

        self._prepare_opt()

        self.frame_pos = np.zeros((self.sample_num, 3))
        self.const_frame_ind = np.array([])

        for c_c in self._cartesian_constraints:

            frame_num, bool_max, c_x, c_y, c_z = c_c

            if frame_num not in self.const_frame_ind:
                self.const_frame_ind = np.append(self.const_frame_ind, frame_num)

        self.frame_traj = np.zeros((len(self.const_frame_ind), self.sample_num, 3))

        print('frames_constrained: {}'.format(self.const_frame_ind))

    def _prepare_opt(self):
        sample_num = 200
        self.sample_num = sample_num

        period = 1.0/self._base_freq
        t = np.linspace(0, period, num=sample_num)

        self.H = np.zeros((self._dyn.dof * sample_num, self._dyn.dyn.n_base))
        self.H_norm = np.zeros((self._dyn.dof * sample_num, self._dyn.dyn.n_base))
    def _obj_func(self, x):
        # objective
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(x)
        # print('q:', q)
        # print('dq: ', dq)
        # print('ddq: ', ddq)

        exec spb.robot_code_to_func( 'python', self._dyn.Hb_code, 'Hb', 'regressor_funcb', self._dyn.rbtdef)
        global sin, cos, sign
        sin = np.sin
        cos = np.cos
        sign = np.sign

        # for n in range(self.sample_num):
        #     vars_input = q[n, :].tolist() + dq[n, :].tolist() + ddq[n, :].tolist()
        #     self.H[n*self._dyn.dof:(n+1)*self._dyn.dof, :] = self._dyn.H_b_func(*vars_input)

        for n in range(self.sample_num):
            self.H[n*self._dyn.dof:(n+1)*self._dyn.dof, :] = np.array( regressor_funcb(q[n, :], dq[n, :], ddq[n, :]) ).reshape(self._dyn.dof,self._dyn.dyn.n_base)
            

        #print('H: ', self.H[n*self._dyn.dof:(n+1)*self._dyn.dof, :])
        self.H /= np.subtract(self.H.max(axis=0), self.H.min(axis=0))

        f = np.linalg.cond(self.H)
        print("cond: {}".format(f))
        # y = self.H
        # xmax, xmin = y.max(), y.min()
        # y = (y - xmin) / (xmax - xmin)
        # #print(y[0,:])
        #
        # f = np.linalg.cond(y)
        # print('f: ', f)

        # constraint
        qIni = np.array([0.0, np.pi/6, 0.0, np.pi/3, 0.0, np.pi/2, 0.0])
        qIniConst = q[0, :] - qIni
        qEndConst = q[-1, :] - qIni
        dqIniConst = dq[0, :]
        dqEndConst = dq[-1, :]
        ddqIniConst = ddq[0, :]
        ddqEndConst = ddq[-1, :]
        EqConst = np.hstack((qIniConst, qEndConst, dqIniConst, dqEndConst, ddqIniConst, ddqEndConst))
        
        g = [0.0] * ((self._const_num * self.sample_num)+EqConst.shape[0])*2
        g_cnt = 0
        
        for i in range(EqConst.shape[0]):
            g[g_cnt] = EqConst[i]
            g_cnt += 1
            g[g_cnt] = -EqConst[i]
            g_cnt += 1
        # Joint constraints (old)
        if len(self._joint_constraints[1]) == 5:
            for j, j_c in enumerate(self._joint_constraints):
                q_s, q_l, q_u, dq_l, dq_u = j_c
                # co_num = self._dyn.coordinates.index(q_s)
            
                for qt, dqt in zip(q[:, j], dq[:, j]):
                    g[g_cnt] = qt - q_u
                    g_cnt += 1
                    g[g_cnt] = q_l - qt
                    g_cnt += 1
                    g[g_cnt] = dqt - dq_u
                    g_cnt += 1
                    g[g_cnt] = dq_l - dqt
                    g_cnt += 1
        else:
            for j, j_c in enumerate(self._joint_constraints):
                q_s, q_l, q_u, dq_l, dq_u, ddq_l, ddq_u = j_c
                # co_num = self._dyn.coordinates.index(q_s)
            
                for qt, dqt, ddqt in zip(q[:, j], dq[:, j], ddq[:, j]):
                    g[g_cnt] = qt - q_u
                    g_cnt += 1
                    g[g_cnt] = q_l - qt
                    g_cnt += 1
                    g[g_cnt] = dqt - dq_u
                    g_cnt += 1
                    g[g_cnt] = dq_l - dqt
                    g_cnt += 1
                    g[g_cnt] = ddqt - ddq_u
                    g_cnt += 1
                    g[g_cnt] = ddq_l - ddqt
                    g_cnt += 1
        fail = 0
        return f, g, fail

    def _add_obj2prob(self):
        self._opt_prob.addObj('f')

    def _add_vars2prob(self):
        joint_coef_num = 2*self._order + 1

        def rand_local(l, u, scale):
            return (np.random.random() * (u - l)/2 + (u + l)/2) * scale

        for num in range(self._dyn.dof):
            # q0
            self._opt_prob.addVar('x'+str(num*joint_coef_num + 1), 'c',
                                  lower=self._q0_min, upper=self._q0_max,
                                  value=rand_local(self._q0_min, self._q0_max, 0.1))
            # a sin
            for o in range(self._order):
                self._opt_prob.addVar('x' + str(num * joint_coef_num + 1 + o + 1), 'c',
                                      lower=self._ab_min, upper=self._ab_max,
                                      value=rand_local(self._ab_min, self._ab_max, 0.1))
            # b cos
            for o in range(self._order):
                self._opt_prob.addVar('x' + str(num * joint_coef_num + 1 + self._order + o + 1), 'c',
                                      lower=self._ab_min, upper=self._ab_max,
                                      value=rand_local(self._ab_min, self._ab_max, 0.1))

    def _add_const2prob(self):
        self._opt_prob.addConGroup('g', self._const_num * self.sample_num + 84, type='i')
        # self._opt_prob.addConGroup('eq', 42, type='e')
        # self._opt_prob.addConGroup('ineq', self._const_num * self.sample_num, type='i')

    def optimize(self):
        #self._prepare_opt()
        self._opt_prob = pyOpt.Optimization('Optimial Excitation Trajectory', self._obj_func)
        self._add_vars2prob()
        self._add_obj2prob()
        self._add_const2prob()

        # print(self._opt_prob)
        #x = np.random.random((self._dyn.rbt_def.dof * (2*self._order+1)))
        #print(self._obj_func(x))

        # PSQP
        # slsqp = pyOpt.pyPSQP.PSQP()
        # slsqp.setOption('MIT', 2)
        # slsqp.setOption('IPRINT', 2)

        # COBYLA
        #slsqp = pyOpt.pyCOBYLA.COBYLA()

        # Genetic Algorithm
        #slsqp = pyOpt.pyNSGA2.NSGA2()

        # SLSQP
        slsqp = pyOpt.pySLSQP.SLSQP()
        slsqp.setOption('IPRINT', 0)
        # slsqp.setOption('MAXIT', 300)
        #slsqp.setOption('ACC', 0.00001)

        # SOLVOPT
        # slsqp = pyOpt.pySOLVOPT.SOLVOPT()
        # slsqp.setOption('maxit', 5)

        #[fstr, xstr, inform] = slsqp(self._opt_prob, sens_type='FD')
        [fstr, xstr, inform] = slsqp(self._opt_prob)

        self.f_result = fstr
        self.x_result = xstr

        print('Condition number: {}'.format(fstr[0]))
        print('x: {}'.format(xstr))
        #print('inform: ', inform)

        print self._opt_prob.solution(0)

    def calc_normalize_mat(self):
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(self.x_result)

        # for n in range(self.sample_num):
        #     vars_input = q[n, :].tolist() + dq[n, :].tolist() + ddq[n, :].tolist()
        #     self.H[n*self._dyn.dof:(n+1)*self._dyn.dof, :] = self._dyn.H_b_func(*vars_input)

        return #self.H.max(axis=0) - self.H.min(axis=0)

    def calc_frame_traj(self):
        q, dq, ddq = self.fourier_traj.fourier_base_x2q(self.x_result)
        # #print(self._dyn.p_n_func[int(self.const_frame_ind[0])])
        # for i in range(len(self.const_frame_ind)):
        #     for num in range(q.shape[0]):
        #         vars_input = q[num, :].tolist()
        #         print(vars_input)

        #         p_num = self._dyn.p_n_func[int(self.const_frame_ind[i])](*vars_input)
        #         #print(p_num[:, 0])
        #         self.frame_traj[i, num, :] = p_num[:, 0]

    def make_traj_csv(self, folder, name, freq, tf):
        x = FourierTraj(self._dyn.dof, self._order, self._base_freq, sample_num_per_period=self._sample_point, frequency=freq, final_time=tf)

        q, dq, ddq = x.fourier_base_x2q(self.x_result)

        with open(folder + name + '.csv', 'wb') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
            for i in range(np.size(q, 0) - 10):
                wr.writerow(np.append(q[i], freq))