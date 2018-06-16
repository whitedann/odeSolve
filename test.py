import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy import sin, cos, sqrt
import matplotlib.animation as animation


class Pendulum():

    def __init__(self,
                 g = 9.8,
                 L = 2.0):
        
        self.init_state = [np.radians(89.0), 0]
        self.time = np.arange(0, 50.0, 0.025) 
        self.g = g
        self.L = L

    def equation(self, y0,t):
        theta, thetaDot = y0
        f = [thetaDot, -(self.g/self.L)*sin(theta)]
        return f

    def equationApprox(self, y0, t):
        theta, thetaDot = y0
        g = [thetaDot, -(self.g/self.L)*theta]
        return g

    def solve_ODE(self):
       
        """Without small-angle approximation"""
        self.state = odeint(self.equation, self.init_state, self.time)
        x1 = sin(self.state[:, 0])*self.L
        y1 = -1*cos(self.state[:, 0])*self.L

        """With small-angle approximation:"""
        self.state2 = odeint(self.equationApprox, self.init_state, self.time)
        x2 = sin(self.state2[:, 0])*self.L
        y2 = -1*cos(self.state2[:, 0])*self.L

        return x1, y1, x2, y2

    def getMax(self):
        return np.degrees(self.init_state[0])


class doublePendulum():

    def __init__(self,
                 g = 9.8,
                 L1 = 2.0,
                 L2 = 1.0,
                 M1 = 1.0,
                 M2 = 5.0):

        """initial state is (theta, z1, phi, z2) in degrees
        where theta is the initial angle of the top rod, z1 the first derivative of theta, z2 is the first deriative of phi """
        self.init_state = [np.radians(120.0), np.radians(0), np.radians(89.0), np.radians(0)]
        self.params = (L1, L2, M1, M2, g)
        self.time = np.arange(0, 50.0, 0.025)

    def equation(self, y0, t):

        (L1, L2, M1, M2, g) = self.params
        theta, thetaDot, phi, phiDot = y0

        delsin = sin(theta - phi)
        delcos = cos(theta - phi)

        dydx = [thetaDot,
                (-M2 * L1 * thetaDot**2 * delsin * delcos + g * M2 * sin(phi) * delcos - M2 * L2 * phiDot**2 * delsin - (M1 + M2) * g * sin(theta))/(L1 * (M1 + M2) - M2 * L1 * delcos**2),
                phiDot,
                (M2 * L2 * phiDot**2 * delsin * delcos + g * sin(theta) * delcos * (M1 + M2) + L1 * thetaDot**2 * delsin * (M1 + M2) - g * sin(phi) * (M1 + M2))/(L2 * (M1 + M2) - M2 * L2 * delcos**2)
                ]
                
        return dydx

    def solve_ODE(self):

        self.state = odeint(self.equation, self.init_state, self.time)
        x1 = sin(self.state[:, 0])*self.params[0]
        y1 = -1*cos(self.state[:, 0])*self.params[0]
        x2 = x1 + sin(self.state[:, 2])*self.params[1]
        y2 = y1 + -1*cos(self.state[:, 2])*self.params[1]

        return x1, y1, x2, y2
       

class coupledPendulum():

    def __init__(self,
                 g = 9.8,
                 L1 = 1.5,
                 L2 = 1.5,
                 M1 = 1.0,
                 M2 = 1.0,
                 k = 0.5):

        """initial state is (theta, thetaDot, phi, phiDot) in degrees
        where theta is the initial angle of the left pendulum, thetaDot is the initial speed
        of the left pendulum, and phi/phiDot is the same for the right pendulum"""
        self.init_state = [np.radians(25.0), 0, np.radians(0.0), 0]
        self.params = (L1, L2, M1, M2, g, k)
        self.time = np.arange(0, 50.0, 0.025)

    def equation(self, y0,t):

        (L1, L2, M1, M2, g, k) = self.params
        theta, thetaDot, phi, phiDot = y0

        dydx = [thetaDot,
                (sin(theta) * (M1 * (L1 * thetaDot * thetaDot - g) - k * L1) + k * L2 * sin(phi)) / (M1 * L1 * cos(theta)),
                phiDot,
                (sin(phi) * (M2 * (L2 * phiDot * phiDot - g) - k * L2) + k * L1 * sin(theta)) / (M2 * L2 * cos(phi))
                ]

        return dydx

    def solve_ODE(self):
        self.state = odeint(self.equation, self.init_state, self.time)

        """convert data into (x,y) coordinates"""
        x1 = sin(self.state[:, 0])*self.params[0]
        y1 = -1*cos(self.state[:, 0])*self.params[0]
        x2 = sin(self.state[:, 2])*self.params[1]
        y2 = -1*cos(self.state[:, 2])*self.params[1]
        return x1, y1, x2, y2

    def getSpringConstant(self):
        return self.params[5]

def initSinglePendulumWindow():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax2.set_xlim(0,50)
    ax2.set_ylim(-1*pend.getMax(),pend.getMax())
    ax2.set_ylabel('Angular Displacement')
    ax2.set_xlabel('time')
    ax2.set_xticklabels(['',10,20,30,40,''])
    fig.subplots_adjust(right=.90, left=.10)
    return line, line2, line4, line5,

def initDoublePendulumWindow():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    ax2.set_visible(False)
    line.set_color("black")
    line2.set_color("black")
    line.set_markerfacecolor("black")
    line2.set_markerfacecolor("black")
    return line, line2, line3

def initCoupledPendulumWindow():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax2.set_xlim(0,50)
    ax2.set_ylim(-45,45)
    line3.set_color("black")
    ax2.set_xticklabels(['',10,20,30,40,''])
    ax2.set_ylabel('Angular Displacment')
    line.set_markerfacecolor("black")
    line2.set_markerfacecolor("black")
    line3.set_markerfacecolor("black")
    return line, line2, line3, line4, line5

pend = Pendulum()
coupPend = coupledPendulum()
dubpend = doublePendulum()

fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111)
ax2 = fig.add_subplot(311)
line, = ax.plot([],[], 'r-', marker='o', lw=2, markersize=10.0, markerfacecolor="red")
line2, = ax.plot([],[], 'b-', marker='o', lw=2, markersize=10.0, markerfacecolor="blue")
line3, = ax.plot([],[])
line4, = ax2.plot([],[], 'r-', lw=1.5)
line5, = ax2.plot([],[], 'b-', lw=1.5)
theta_text = ax2.text(0.30, 0.03, 'Phase difference = ', transform = ax.transAxes, fontsize=15)
time_text = ax.text(0.03, 0.03, 'time = 0s', transform = ax.transAxes, fontsize=15)
xdata, ydata1, ydata2 = [], [], []
singlePendulumData = np.array(pend.solve_ODE())
doublePendulumData = np.array(dubpend.solve_ODE())
coupledPendulumData = np.array(coupPend.solve_ODE())

#Animation functions for single, double and coupled pendulums

def animateCoupledPendulum(num,data2,line,line2, line3, line4, line5):
    line.set_data([-1, data2[0, num] - 1], [0, data2[1, num]])
    line2.set_data([1, data2[2, num] + 1], [0, data2[3, num]])
    line3.set_data([data2[0, num] -1, data2[2, num] + 1], [data2[1, num], data2[3, num]])
    
    xdata.append(num/40.0)
    ydata1.append(np.degrees(np.arctan(data2[0,num]/data2[1,num])))
    ydata2.append(np.degrees(np.arctan(data2[2,num]/data2[3,num])))

    line4.set_data(xdata, ydata1)
    line5.set_data(xdata, ydata2)
    
    theta_text.set_text('Spring Tension: ' + str(abs(round(coupPend.getSpringConstant()*(data2[2,num]-data2[0,num]),1))) + 'N')
    time_text.set_text('t= ' + str(round(num*0.025,1)) + 's')
    return line, line2, line3, line4, line5

def animateSinglePendulum(num, data1, line, line2):
    """draw the two pendulums as lines 1 and 2"""
    line.set_data([0,data1[0,num]], [0,data1[1,num]])
    line2.set_data([0,data1[2,num]], [0,data1[3,num]])

    """(xdata, ydata) is the (time, angular displacement) of the displacement plot""" 
    """the time data is the same for both pendulums, so xdata is used both times"""
    xdata.append(num/40.0)
    ydata1.append(np.degrees(np.arctan(data1[0,num]/data1[1,num])))
    ydata2.append(np.degrees(np.arctan(data1[2,num]/data1[3,num])))
    
    """line 4 is the plot without the small-angle approximation. line 5 is the plot with the small-angle approximation""" 
    line4.set_data(xdata,ydata1)
    line5.set_data(xdata,ydata2)

    """Update the data for the time and phase difference text"""
    theta_text.set_text('Phase difference = ' + str(int(ydata2[num]-ydata1[num])) + ' degrees')
    time_text.set_text('t = ' + str(round(num*0.025, 1)) + 's')

    return line, line2

def animateDoublePendulum(num, data3, line, line2, line3):
    line.set_data([0,data3[0,num]], [0,data3[1,num]])
    line2.set_data([data3[0,num],data3[2,num]], [data3[1,num],data3[3,num]])
    
    xdata.append(data3[2,num])
    ydata1.append(data3[3,num])
    line3.set_data(xdata, ydata1)
    time_text.set_text('t = ' + str(round(num*0.025, 1)) + 's')
    return line, line2, line3

#ani = animation.FuncAnimation(fig, animateSinglePendulum, interval=1, frames=2000, fargs=(singlePendulumData, line, line2), init_func=initSinglePendulumWindow, repeat=False)

#ani = animation.FuncAnimation(fig, animateCoupledPendulum, interval=1, frames=2000, fargs=(coupledPendulumData, line, line2, line3, line4, line5), init_func=initCoupledPendulumWindow, repeat=False)

ani = animation.FuncAnimation(fig, animateDoublePendulum, interval=1, frames=2000, fargs=(doublePendulumData, line, line2, line3), init_func=initDoublePendulumWindow, repeat=False)

#plt.show()

ani.save('dPend89.mp4', fps=40, dpi=300)












