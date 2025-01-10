import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import pyqtgraph.parametertree as ptree
from scipy.optimize import minimize
import numpy as np
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
import random
from utils import *
import lcm
import sys




class ImpTune(QtWidgets.QMainWindow):
    def __init__(self, dataset='t'):
        super().__init__()
        self.resize(800, 1000)
        self.layout = pg.LayoutWidget()
        self.setCentralWidget(self.layout)

        self.dataset = dataset
        self.context = [1.0, 0]
        
        if dataset == "t":
            from opensource_loader import load_data
            from opensource_loader import v_key, s_key
            self.m = 75
            data = load_data(vk=str(self.context[0]),sk=str(self.context[1]),m=self.m)

        elif dataset == "g":
            from greggdataset_loader import load_data
            from greggdataset_loader import v_key, s_key
            self.m = 75
            _, data = load_data(ab_k="AB06",vk=str(self.context[0]),sk=str(self.context[1]),m=self.m)

        self.dt = grid_interp_dt(v=self.context[0], s=self.context[1], num_frame=data.shape[1])
        self.q_k = data[0]
        self.t_k = data[1]
        self.q_a = data[2]
        self.t_a = data[3]
        self.t_k_pred1 = np.zeros((100,))
        self.t_a_pred1 = np.zeros((100,))
        self.t_k_pred2 = np.zeros((100,))
        self.t_a_pred2 = np.zeros((100,))
        self.t_k_pred3 = np.zeros((100,))
        self.t_a_pred3 = np.zeros((100,))
        self.t_k_pred4 = np.zeros((100,))
        self.t_a_pred4 = np.zeros((100,))
        self.idx = np.arange(self.q_k.shape[0])
        self.lim_kkp = [[0,5], [0,5], [0,1.2], [0,1.2]]
        self.lim_kkb = [[0,1], [0,1], [0,1], [0,1]]
        self.lim_kqe = [[0,25], [0,25], [0,80], [0,30]]
        self.lim_akp = [[0,5], [0,5], [0,1], [0,1]]
        self.lim_akb = [[0,1], [0,1], [0,1], [0,1]]
        self.lim_aqe = [[-15,15], [0,12], [-5,10], [-15,15]]
        
        self.k_mat = [[0,0,0,0],[0,0,0,0]]
        self.b_mat = [[0,0,0,0],[0,0,0,0]]
        self.q_e_mat = [[0,0,0,0],[0,0,0,0]]

        self.tp = 0 # chosen phase
        self.gait_divisions = []

        self.win = pg.GraphicsLayoutWidget(show=True)
        self.init_win()
        self.init_slider()
        self.set_slider_pos()

        tree = ParameterTree()
        vtype = dict(type="list", values=v_key)
        stype = dict(type="list", values=s_key)
        contextParam = interact(self.next_data,v_k=vtype,s_k=stype)
        tree.setMinimumWidth(200)
        tree.addParameters(contextParam, showTop=True)

        sendParam = interact(self.send_imp)
        tree.addParameters(sendParam, showTop=True)

        saveParam = interact(self.save_imp)
        tree.addParameters(saveParam, showTop=True)

        self.params = ptree.Parameter.create(name='Parameters', type='group', children=[
            dict(name='phase',type='int', value=0, limits=[0,3])
        ])
        self.params.children()[0].sigValueChanged.connect(self.change_tp)
        tree.addParameters(self.params)

        self.layout.addWidget(tree, row=0, col=0)
        self.layout.addWidget(self.win, row=3, col=0, rowspan=2,colspan=5)
        self.lc = lcm.LCM()


    def init_win(self):
        p0 = self.win.addPlot()
        p0.setFixedWidth(300)
        p0.setYRange(-20, 50)
        p1 = self.win.addPlot()
        p1.setFixedWidth(300)
        p1.setYRange(-10,100)
        self.win.nextRow()
        p2 = self.win.addPlot()
        p2.setFixedWidth(300)
        p2.setYRange(-20, 80)
        p3 = self.win.addPlot()
        p3.setFixedWidth(300)
        p3.setYRange(-10,100)
        self.ax_kqt = p0.plot(self.q_k, self.t_k)
        self.ax_aqt = p1.plot(self.q_a, self.t_a)
        self.ax_kq = p2.plot(self.idx, self.q_k)
        self.ax_aq = p3.plot(self.idx, self.q_a)
        self.ax_kt = p2.plot(self.idx, self.t_k)
        self.ax_at = p3.plot(self.idx, self.t_a)
        self.ax_kqt_phase = [
            p0.plot(self.q_k, self.t_k, pen=pg.mkPen(color=(255,0,0,150), width=3)),
            p0.plot(self.q_k, self.t_k, pen=pg.mkPen(color=(0,255,0,150), width=3)),
            # p0.plot(self.q_k, self.t_k, pen=pg.mkPen(color=(0,0,255,150), width=3)),
            # p0.plot(self.q_k, self.t_k, pen=pg.mkPen(color=(255,0,255,150), width=3))
        ]
        self.ax_aqt_phase = [
            p1.plot(self.q_a, self.t_a, pen=pg.mkPen(color=(255,0,0,150), width=3)),
            p1.plot(self.q_a, self.t_a, pen=pg.mkPen(color=(0,255,0,150), width=3)),
            # p1.plot(self.q_a, self.t_a, pen=pg.mkPen(color=(0,0,255,150), width=3)),
            # p1.plot(self.q_a, self.t_a, pen=pg.mkPen(color=(255,0,255,150), width=3))
        ]
        self.ax_kq_phase = [
            p2.plot(self.idx, self.t_k, pen=pg.mkPen(color=(255,0,0,150), width=5)),
            p2.plot(self.idx, self.t_k, pen=pg.mkPen(color=(0,255,0,150), width=5)),
            p2.plot(self.idx, self.t_k, pen=pg.mkPen(color=(0,0,255,150), width=5)),
            p2.plot(self.idx, self.t_k, pen=pg.mkPen(color=(255,0,255,150), width=5))
        ]
        self.ax_aq_phase = [
            p3.plot(self.idx, self.t_a, pen=pg.mkPen(color=(255,0,0,150), width=5)),
            p3.plot(self.idx, self.t_a, pen=pg.mkPen(color=(0,255,0,150), width=5)),
            p3.plot(self.idx, self.t_a, pen=pg.mkPen(color=(0,0,255,150), width=5)),
            p3.plot(self.idx, self.t_a, pen=pg.mkPen(color=(255,0,255,150), width=5))
        ]
        # self.ax_fswing_phase4 = p2.plot(self.idx, self.t_a)
        self.init_curve()
    
    def init_curve(self):
        self.ax_kqt.setData(x=self.q_k, y=self.t_k)
        self.ax_aqt.setData(x=self.q_a, y=self.t_a)
        self.ax_kq.setData(y=self.q_k)
        self.ax_aq.setData(y=self.q_a)
        self.ax_kt.setData(y=self.t_k)
        self.ax_at.setData(y=self.t_a)
        imp_calculator = IJRR_Imp_Cal(qk=self.q_k, tk=self.t_k, 
                                      qa=self.q_a, ta=self.t_a, dt=self.dt,
                                      impTune=self, headless=False)
        imp_calculator.cal_imp_IJRR()
        self.print_imp()
        
    def print_imp(self):
        k_mat_used = np.round(np.array(self.k_mat)*30,2).tolist()
        b_mat_used = np.round(np.array(self.b_mat)*30,2).tolist()
        qe_mat_used = np.round(np.array(self.q_e_mat),2).tolist()

        print("k_mat[0]=", k_mat_used)
        print("b_mat[0]=", b_mat_used)
        print("q_e_mat[0]=", qe_mat_used)

    def init_slider(self):

        p = pg.PlotWidget(name='knee_kp')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=0, col=2)
        self.line_kkp = pg.InfiniteLine(angle=90, label='kp={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(255,0,0,150), width=10))
        p.addItem(self.line_kkp)
        self.line_kkp.setBounds([self.lim_kkp[self.tp][0],self.lim_kkp[self.tp][1]])
        p.setXRange(0,10)
        self.line_kkp.sigDragged.connect(self.updateline_kkp)

        p = pg.PlotWidget(name='knee_kb')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=0, col=3)
        self.line_kkb = pg.InfiniteLine(angle=90,label='kb={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(0,255,0,150), width=10))
        p.addItem(self.line_kkb)
        self.line_kkb.setBounds([self.lim_kkb[self.tp][0],self.lim_kkb[self.tp][1]])
        p.setXRange(0,1)
        self.line_kkb.sigDragged.connect(self.updateline_kkb)


        p = pg.PlotWidget(name='knee_qe')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=0, col=4)
        self.line_kqe = pg.InfiniteLine(angle=90,label='qe={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(0,0,255,150), width=10))
        p.addItem(self.line_kqe)
        self.line_kqe.setBounds([self.lim_kqe[self.tp][0], self.lim_kqe[self.tp][1]])
        p.setXRange(0,90)
        self.line_kqe.sigDragged.connect(self.updateline_kqe)

        p = pg.PlotWidget(name='ankle_kp')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=1, col=2)
        self.line_akp = pg.InfiniteLine(angle=90,label='kp={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(255,0,0,150), width=10))
        p.addItem(self.line_akp)
        self.line_akp.setBounds([self.lim_akp[self.tp][0], self.lim_akp[self.tp][1]])
        p.setXRange(0,10)
        self.line_akp.sigDragged.connect(self.updateline_akp)


        p = pg.PlotWidget(name='ankle_kb')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=1, col=3)
        self.line_akb = pg.InfiniteLine(angle=90,label='kb={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(0,255,0,150), width=10))
        p.addItem(self.line_akb)
        self.line_akb.setBounds([self.lim_akb[self.tp][0], self.lim_akb[self.tp][1]])
        p.setXRange(0,1)
        self.line_akb.sigDragged.connect(self.updateline_akb)


        p = pg.PlotWidget(name='ankle_qe')
        p.setFixedSize(200,100)
        self.layout.addWidget(p, row=1, col=4)
        self.line_aqe = pg.InfiniteLine(angle=90,label='qe={value:0.2f}',
                                        movable=True,pen=pg.mkPen(color=(0,0,255,150), width=10))
        p.addItem(self.line_aqe)
        self.line_aqe.setBounds([self.lim_aqe[self.tp][0], self.lim_aqe[self.tp][1]])
        p.setXRange(-10,25)
        self.line_aqe.sigDragged.connect(self.updateline_aqe)
        
    def set_slider_pos(self):
        self.line_kkp.setPos(pos=self.k_mat[0][self.tp])
        self.line_kkb.setPos(pos=self.b_mat[0][self.tp])
        self.line_kqe.setPos(pos=self.q_e_mat[0][self.tp])
        self.line_akp.setPos(pos=self.k_mat[1][self.tp])
        self.line_akb.setPos(pos=self.b_mat[1][self.tp])
        self.line_aqe.setPos(pos=self.q_e_mat[1][self.tp])

    def updateline_kkp(self):
        self.line_kkp.setBounds([self.lim_kkp[self.tp][0],self.lim_kkp[self.tp][1]])
        self.k_mat[0][self.tp] = self.line_kkp.value()
        self.replot_kt()
        self.print_imp()
    
    def updateline_kkb(self):
        self.line_kkb.setBounds([self.lim_kkb[self.tp][0],self.lim_kkb[self.tp][1]])
        self.b_mat[0][self.tp] = self.line_kkb.value()
        self.replot_kt()
        self.print_imp()

    def updateline_kqe(self):
        self.line_kqe.setBounds([self.lim_kqe[self.tp][0],self.lim_kqe[self.tp][1]])
        if self.tp <= 1:
            self.q_e_mat[0][self.tp] = self.line_kqe.value()
            self.replot_kt()
            self.print_imp()
    
    def updateline_akp(self):
        self.line_akp.setBounds([self.lim_akp[self.tp][0],self.lim_akp[self.tp][1]])
        self.k_mat[1][self.tp] = self.line_akp.value()
        self.replot_at()
        self.print_imp()

    def updateline_akb(self):
        self.line_akb.setBounds([self.lim_akb[self.tp][0],self.lim_akb[self.tp][1]])
        self.b_mat[1][self.tp] = self.line_akb.value()
        self.replot_at()
        self.print_imp()
    
    def updateline_aqe(self):
        self.line_aqe.setBounds([self.lim_aqe[self.tp][0],self.lim_aqe[self.tp][1]])
        self.q_e_mat[1][self.tp] = self.line_aqe.value()
        self.replot_at()
        self.print_imp()

    def next_data(self, v_k="1.0", s_k="0"):
        if self.dataset == "t":
            from opensource_loader import load_data
            data = load_data(v_k, s_k, m=self.m)
        else:
            from greggdataset_loader import load_data
            _, data = load_data(v_k, s_k, m=self.m)
        if data is not None:
            self.context[0] = float(v_k)
            self.context[1] = float(s_k)
            self.dt = grid_interp_dt(v=self.context[0], s=self.context[1], num_frame=data.shape[1])
            self.q_k = data[0]
            self.t_k = data[1]
            self.q_a = data[2]
            self.t_a = data[3]
            self.init_curve()
            self.set_slider_pos()
    
    def change_tp(self):
        self.tp = self.params.child('phase').value()
        self.set_slider_pos()

    def send_imp(self):
        k_mat_used = np.round(np.array(self.k_mat)*30,2)
        b_mat_used = np.round(np.array(self.b_mat)*30,2)
        qe_mat_used = np.round(np.array(self.q_e_mat),2)
        para_send = np.zeros((4,6))
        para_send[:,0] = k_mat_used[0,:]
        para_send[:,1] = b_mat_used[0,:]
        para_send[:,2] = qe_mat_used[0,:]
        para_send[:,3] = k_mat_used[1,:]
        para_send[:,4] = b_mat_used[1,:]
        para_send[:,5] = qe_mat_used[1,:]
        print("This Version can't send imp")
        # msg = impedance_info()
        # msg.para_phase1 = para_send[0,:]
        # msg.para_phase2 = para_send[1,:]
        # msg.para_phase3 = para_send[2,:]
        # msg.para_phase4 = para_send[3,:]
        # self.lc.publish("Impedance_Info", msg.encode())
    
    def save_imp(self):
        k_mat_used = np.round(np.array(self.k_mat)*30,2)
        b_mat_used = np.round(np.array(self.b_mat)*30,2)
        qe_mat_used = np.round(np.array(self.q_e_mat),2)
        para_send = np.zeros((4,6))
        para_send[:,0] = k_mat_used[0,:]
        para_send[:,1] = b_mat_used[0,:]
        para_send[:,2] = qe_mat_used[0,:]
        para_send[:,3] = k_mat_used[1,:]
        para_send[:,4] = b_mat_used[1,:]
        para_send[:,5] = qe_mat_used[1,:]
        print("This version can't save")
        # save_path = "/home/yuxuan/Project/HPS_Perception/map_ws/src/HPS_Perception/hps_baseline/result/"
        # np.save(save_path+"v_{}_s_{}_best.npy".format(self.context[0], self.context[1]), para_send)

    def replot_kt(self):
        p_0, p_1 = self.gait_divisions[self.tp][0], self.gait_divisions[self.tp][1]
        idx = np.arange(p_0, p_1)
        k, b, qe = self.k_mat[0][self.tp],self.b_mat[0][self.tp], self.q_e_mat[0][self.tp] 
        t_k_pred = k*(self.q_k[idx]-qe)+b*self.dq_k[idx]
        self.ax_kq_phase[self.tp].setData(x=idx, y=t_k_pred)

    def replot_at(self):
        p_0, p_1 = self.gait_divisions[self.tp][0], self.gait_divisions[self.tp][1]
        idx = np.arange(p_0, p_1)
        k, b, qe = self.k_mat[1][self.tp],self.b_mat[1][self.tp], self.q_e_mat[1][self.tp] 
        t_a_pred = k*(self.q_a[idx]+qe)+b*self.dq_a[idx]
        self.ax_aq_phase[self.tp].setData(x=idx, y=t_a_pred)



class IJRR_Imp_Cal:
    def __init__(self, qk, tk, qa, ta, dt, impTune:ImpTune=None, headless=False):
        self.q_k = qk
        self.t_k = tk
        self.q_a = qa
        self.t_a = ta
        self.headless = headless
        if not headless:
            self.impTune = impTune
        self.idx = np.arange(self.q_k.shape[0])
        self.dt = dt
        self.k_mat = [[0,0,0,0],[0,0,0,0]]
        self.b_mat = [[0,0,0,0],[0,0,0,0]]
        self.q_e_mat = [[0,0,0,0],[0,0,0,0]]
    
    def reinit(self, qk, tk, qa, ta, dt):
        self.q_k = qk
        self.t_k = tk
        self.q_a = qa
        self.t_a = ta
        self.idx = np.arange(self.q_k.shape[0])
        self.dt = dt
        self.k_mat = [[0,0,0,0],[0,0,0,0]]
        self.b_mat = [[0,0,0,0],[0,0,0,0]]
        self.q_e_mat = [[0,0,0,0],[0,0,0,0]]

    def gait_division_IJRR(self):
        gait_divisions = []
        idx_max_qa, max_qa = np.argmax(self.q_a[:50]), np.max(self.q_a[:50])
        qa_threshold = max_qa - 2
        gait_divisions.append([0, np.where(self.q_a[:idx_max_qa]>qa_threshold)[0][0]])
        idx_max_ta, max_ta = np.argmax(self.t_a), np.max(self.t_a)
        ta_threshold = 20
        gait_divisions.append([gait_divisions[-1][1], np.where(self.t_a[idx_max_ta:80]<ta_threshold)[0][0]+idx_max_ta])
        gait_divisions.append([gait_divisions[-1][1], np.argmax(self.q_k)])
        gait_divisions.append([gait_divisions[-1][1], self.idx[-1]+1])
        self.gait_divisions = gait_divisions
    
    def cal_st_qe_MT(self):
        def staticimpFun(x, k, qe):
            return k*(x-qe)
        def plot_result(p_1, imp, q, t, ax):
            if q[p_1] < imp[1]:
                if imp[1]-q[p_1] < 5:
                    q_ = np.linspace(q[p_1]-2,q[p_1]+2, 10)
                else:
                    q_ = np.linspace(q[p_1],imp[1], 10)
            else:
                if q[p_1]-imp[1] < 5:
                    q_ = np.linspace(q[p_1]-2,q[p_1]+2, 10)
                else:
                    q_ = np.linspace(imp[1], q[p_1],10)
            t_pred = staticimpFun(q_, imp[0], imp[1])
            ax.setData(x=q_, y=t_pred)
        
        self.qe_k = [0, 0, 0, 0]
        self.qe_a = [0, 0, 0, 0]

        p, c = 0, 'r'
        imp_k, p_0, p_1, p_2 = cal_qe_knee_phase1(self.q_k, self.t_k)
        if not self.headless:
            plot_result(p_1, imp_k, self.q_k, self.t_k, ax=self.impTune.ax_kqt_phase[p])
        self.qe_k[0] = imp_k[1]
        self.k_mat[0][p] = imp_k[0] if imp_k[0]<6 else 6

        p, c = 1, 'g'
        imp_k, p_0, p_1, p_2 = cal_qe_knee_phase2(self.q_k, self.t_k)
        if not self.headless:
            plot_result(p_1, imp_k, self.q_k, self.t_k, ax=self.impTune.ax_kqt_phase[p])
        self.qe_k[1] = imp_k[1]
        self.k_mat[0][p] = imp_k[0] if imp_k[0]<6 else 6

        p, c = 0, 'r'
        imp_a, p_0, p_1, p_2 = cal_qe_ankle_phase1(self.q_a, self.t_a)
        if not self.headless:
            plot_result(p_1, imp_a, self.q_a, self.t_a, ax=self.impTune.ax_aqt_phase[p])
        self.qe_a[0] = imp_a[1]
        self.k_mat[1][p] = imp_a[0] if imp_a[0]<8 else 8

        p, c = 1, 'g'
        imp_a, p_0, p_1, p_2 = cal_qe_ankle_phase2(self.q_a, self.t_a)
        if not self.headless:
            plot_result(p_1, imp_a, self.q_a, self.t_a, ax=self.impTune.ax_aqt_phase[p])
        self.qe_a[1] = imp_a[1]
        self.k_mat[1][p] = imp_a[0] if imp_a[0]<8 else 8
        
    def cal_sw_imp_YX(self):
        q_k_swing = self.q_k[self.gait_divisions[1][1]:self.gait_divisions[3][1]]
        q_opt_s1 = []
        dq_opt_s1 = []
        t_opt_s1 = []
        def cost_function(params, qdes, dt=0.01, degrees=True):
            nonlocal q_opt_s1, dq_opt_s1, t_opt_s1
            q_opt_s1 = []
            dq_opt_s1 = []
            t_opt_s1 = []
            kp1, kb1, kp2, kb2 = params
            if degrees:
                qdes = np.deg2rad(qdes)
            qe1 = np.max(qdes)+np.deg2rad(10)
            s1 = np.argmax(qdes)
            q = qdes[0]
            dq = 0
            total_error = 0
            qk_buf = np.zeros((5,))
            for i in range(s1+5):
                qk_buf[0:-1] = qk_buf[1:]
                qk_buf[-1] = q
                q_opt_s1.append(q)
                dq_opt_s1.append(dq)
                t_opt_s1.append(dt*i)
                refq = qdes[i]
                e = q-refq
                if qe1 - np.min(qk_buf) > np.deg2rad(10):
                    u = -kp1*(q-qe1)-kb1*dq
                else:
                    break
                q, dq = system_model(q, dq, u, dt)
                total_error += e**2
            return total_error
        q_opt_s2 = []
        dq_opt_s2 = []
        t_opt_s2 = []
        def cost_function2(params, qdes, dt=0.01, degress=True):
            nonlocal q_opt_s2, dq_opt_s2, t_opt_s2
            kp2, kb2 = params
            q_opt_s2 = []
            dq_opt_s2 = []
            t_opt_s2 = []
            if degress:
                qdes = np.deg2rad(qdes)
            qe2 = qdes[-1]
            s2 = np.shape(qdes)[0]
            q = q_opt_s1[-1]
            dq = dq_opt_s1[-1]
            total_error = 0
            for i in range(len(t_opt_s1), s2):
                q_opt_s2.append(q)
                dq_opt_s2.append(dq)
                t_opt_s2.append(i*dt)
                refq = qdes[i]
                e = q-refq
                u = -kp2*(q-qe2)-kb2*dq
                q, dq = system_model(q, dq, u, dt)
                total_error += 0.05*e**2
            total_error += 10*(abs(qdes[-1]-q))**2
            return total_error
        def plot_q_opt(ax_kq_phase):
            ax_kq_phase[2].setData(x=np.arange(self.gait_divisions[2][0], 
                                                    self.gait_divisions[2][0]+len(q_opt_s1)), 
                                        y=np.rad2deg(q_opt_s1))
            ax_kq_phase[3].setData(x=np.arange(self.gait_divisions[2][0]+len(q_opt_s1), 
                                                    self.gait_divisions[2][0]+len(q_opt_s1)+len(q_opt_s2)), 
                                        y=np.rad2deg(q_opt_s2))

        q_k_swing, t_swing = get_parabola(
                q0=q_k_swing[0],q1=np.max(q_k_swing),q2=q_k_swing[-1],
                s1=np.argmax(q_k_swing),s2=np.shape(q_k_swing)[0]-1,dt=self.dt)
        idx_swing_max = np.argmax(q_k_swing)
        
        result = minimize(cost_function, (20,3,20,3), args=(q_k_swing, self.dt), method='SLSQP',
                            bounds=((20,60),(np.rad2deg(1.5/30),np.rad2deg(2.5/30)),
                                    (20,60),(np.rad2deg(2/30),np.rad2deg(2.5/30))))
        # result = minimize(cost_function, (20,1,20,1), args=(q_k_swing, self.dt), method='SLSQP',
        #                     bounds=((20,50),(1.3,2),(20,50),(1,2)))
        kp3, kb3, _,_ = result.x
        # print("kp, kb, swing1:", kp3, kb3)
        qe3 = np.max(q_k_swing)+10
        imp_k3 = [np.deg2rad(kp3), np.deg2rad(kb3), qe3]
        result2 = minimize(cost_function2, (20,3), args=(q_k_swing, self.dt), method='SLSQP',
                            bounds=((20,60),(np.rad2deg(2/30),np.rad2deg(2.5/30))))
        # result2 = minimize(cost_function2, (20,1), args=(q_k_swing, self.dt), method='SLSQP',
        #                     bounds=((20,50),(0.5,2)))
        kp4, kb4 = result2.x
        # qe4 = self.qe_k[0]
        qe4 = q_k_swing[-1] if q_k_swing[-1] > 2 else 2
        imp_k4 = [np.deg2rad(kp4), np.deg2rad(kb4), qe4]
        if not self.headless:
            plot_q_opt(self.impTune.ax_kq_phase)
        
        return imp_k3, imp_k4

    def cal_imp_IJRR(self, undamping=False):
        q_k, q_a, t_k, t_a = self.q_k, self.q_a, self.t_k, self.t_a
        self.gait_division_IJRR()
        dq_k = np.diff(np.append(q_k[-1], q_k))/self.dt
        dq_k[0] = dq_k[1]
        dq_a = np.diff(np.append(q_a[-1], q_a))/self.dt
        dq_a[0] = dq_a[1]
        self.dq_k = dq_k
        self.dq_a = dq_a
        k_mat = [[0,0,0,0],[0,0,0,0]]
        b_mat = [[0,0,0,0],[0,0,0,0]]
        q_e_mat = [[0,0,0,0],[0,0,0,0]]
        
        self.cal_st_qe_MT()
        qe_k, qe_a = np.array(self.qe_k), -np.array(self.qe_a)

        b_stance = 10/30

        p, c = 0, 'r'
        p_0, p_1 = self.gait_divisions[p][0], self.gait_divisions[p][1]
        idx = np.arange(p_0, p_1)
        if undamping:
            imp_k, t_k_pred = cal_imp_kbqe(q_k[idx],dq_k[idx],t_k[idx], bound=0, unbound=True)
            imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=0, unbound=True)
        else:
            # bound_k = ((2,0.01,qe_k[p]-3),(3,0.05,qe_k[p]+2))
            # bound_a = ((0,0), (3,0.2))
            # imp_k, t_k_pred = cal_imp_kbqe(q_k[idx],dq_k[idx],t_k[idx], bound=bound_k, unbound=False)
            # imp_a, t_a_pred = cal_imp_kb(q_a[idx],dq_a[idx],qe_a[p],t_a[idx], bound=bound_a, 
            #                              unbound=False, reverse=True)
            imp_k = [self.k_mat[0][0], b_stance, qe_k[0]]
            t_k_pred = cal_tpred(q_k[idx], dq_k[idx], imp_k, reverse=False)
            imp_a = [self.k_mat[1][0], b_stance, qe_a[0]]
            t_a_pred = cal_tpred(q_a[idx], dq_a[idx], imp_a, reverse=True)

        def plot_result(ax_kq_phase, ax_aq_phase):
            if p<=1:
                ax_kq_phase[p].setData(x=idx, y=t_k_pred)
            ax_aq_phase[p].setData(x=idx, y=t_a_pred)
        
        if not self.headless:
            plot_result(self.impTune.ax_kq_phase, self.impTune.ax_aq_phase)
        imp_k, imp_a = np.round(imp_k, 3), np.round(imp_a, 3)
        k_mat[0][p], b_mat[0][p], q_e_mat[0][p] = imp_k[0], imp_k[1], imp_k[2]
        k_mat[1][p], b_mat[1][p], q_e_mat[1][p] = imp_a[0], imp_a[1], imp_a[2]

        p, c = 1, 'g'
        p_0, p_1 = self.gait_divisions[p][0], self.gait_divisions[p][1]
        idx = np.arange(p_0, p_1)
        if undamping:
            imp_k, t_k_pred = cal_imp_kbqe(q_k[idx],dq_k[idx],t_k[idx], bound=0, unbound=True)
            imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=0, unbound=True)
        else:
            # bound_k = ((2,0.05,qe_k[p]-3),(3,0.1,qe_k[p]+3))
            # bound_a = ((0,0.05), (3,0.1))
            # idx_max_ta = np.argmax(t_a)
            # imp_k, t_k_pred = cal_imp_kbqe(q_k[idx],dq_k[idx],t_k[idx], bound=bound_k, unbound=False)
            # # imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=bound_a, unbound=False)
            # imp_a, t_a_pred = cal_imp_kb(q_a[idx], dq_a[idx], qe_a[p], t_a[idx], bound=bound_a, 
            #                              unbound=False, reverse=True)
            imp_k = [self.k_mat[0][1], b_stance, qe_k[1]]
            t_k_pred = cal_tpred(q_k[idx], dq_k[idx], imp_k, reverse=False)
            imp_a = [self.k_mat[1][1], b_stance, qe_a[1]]
            t_a_pred = cal_tpred(q_a[idx], dq_a[idx], imp_a, reverse=True)

        if not self.headless:
            plot_result(self.impTune.ax_kq_phase, self.impTune.ax_aq_phase)
        imp_k, imp_a = np.round(imp_k, 3), np.round(imp_a, 3)
        k_mat[0][p], b_mat[0][p], q_e_mat[0][p] = imp_k[0], imp_k[1], imp_k[2]
        k_mat[1][p], b_mat[1][p], q_e_mat[1][p] = imp_a[0], imp_a[1], imp_a[2]
        
        imp_k3, imp_k4 = self.cal_sw_imp_YX()

        p, c = 2, 'b'
        p_0, p_1 = self.gait_divisions[p][0], self.gait_divisions[p][1]
        idx = np.arange(p_0, p_1)
        imp_k = imp_k3
        if undamping:
            imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=0, unbound=True)
        else:
            bound_a = ((0.7,0.025,-5), (0.8,0.03,-2))
            imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=bound_a, 
                                           unbound=False, reverse=True)
        if not self.headless:
            plot_result(self.impTune.ax_kq_phase, self.impTune.ax_aq_phase)
        imp_k, imp_a = np.round(imp_k, 3), np.round(imp_a, 3)
        k_mat[0][p], b_mat[0][p], q_e_mat[0][p] = imp_k[0], imp_k[1], imp_k[2]
        k_mat[1][p], b_mat[1][p], q_e_mat[1][p] = imp_a[0], imp_a[1], imp_a[2]

        p, c = 3, 'm'
        p_0, p_1 = self.gait_divisions[p][0], self.gait_divisions[p][1]
        idx = np.arange(p_0, p_1)
        imp_k = imp_k4
        if undamping:
            imp_a, t_a_pred = cal_imp_kbqe(q_a[idx],dq_a[idx],t_a[idx], bound=0, unbound=True)
        else:
            bound_a = ((0.6,0.03), (0.7,0.031))
            imp_a, t_a_pred = cal_imp_kb(q_a[idx],dq_a[idx],qe_a[0],t_a[idx], bound=bound_a, 
                                         unbound=False, reverse=True)
        if not self.headless:
            plot_result(self.impTune.ax_kq_phase, self.impTune.ax_aq_phase)
        imp_k, imp_a = np.round(imp_k, 3), np.round(imp_a, 3)
        k_mat[0][p], b_mat[0][p], q_e_mat[0][p] = imp_k[0], imp_k[1], imp_k[2]
        k_mat[1][p], b_mat[1][p], q_e_mat[1][p] = imp_a[0], imp_a[1], imp_a[2]

        self.k_mat = k_mat
        self.b_mat = b_mat
        self.q_e_mat = q_e_mat

        if not self.headless:
            self.impTune.gait_divisions = self.gait_divisions
            self.impTune.qe_k = self.qe_k
            self.impTune.qe_a = self.qe_a
            self.impTune.k_mat = self.k_mat
            self.impTune.b_mat = self.b_mat
            self.impTune.q_e_mat = self.q_e_mat
        
    def cal_gait_fea(self):
        phase = np.linspace(0,1,self.q_k.shape[0])
        t = phase*self.dt
        P3, p_P3 = np.max(self.q_k[50:]), np.argmax(self.q_k[50:])+50
        P4, p_P4 = np.min(self.q_k[80:]), np.argmin(self.q_k[80:])+80
        P2, p_P2 = np.min(self.q_k[30:50]), np.argmin(self.q_k[30:40])+30
        P1, p_P1 = np.max(self.q_k[10:30]), np.argmax(self.q_k[10:15])+10
        return np.array([P1, P2, P3, P4, p_P1, p_P2, p_P3, p_P4]).reshape((-1,))
    