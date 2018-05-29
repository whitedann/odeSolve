import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy import sin, cos, sqrt
import matplotlib.animation as animation

class Pendulum():

    def __init__(self):
        self.l = 2.0 
        self.g = 9.8
        self.k = 2.5 
        self.m = 1.0
        self.init_state = [np.radians(45.0), 0]
        self.init_state2 = [np.radians(45.0), 0, np.radians(4.0), 0]
        self.init_state3 = [np.radians(90.0), 0, np.radians(10.0), 0]
        self.time = np.arange(0, 50.0, 0.025)


    def equation(self, y0,t):
        theta, thetaDot = y0
        f = [thetaDot, -(self.g/self.l)*sin(theta)]
        return f

    def equation2(self, y0, t):
        """""state=(theta, thetaDot, phi, phiDot)"""
        theta, thetaDot, phi, phiDot = y0
        dydx = [thetaDot, (sin(theta)*(self.m*(self.l*thetaDot*thetaDot-self.g)-self.k*self.l)+self.k*self.l*sin(phi))/(self.m*self.l*cos(theta)) ,
                phiDot, (sin(phi)*(self.m*(self.l*phiDot*phiDot-self.g)-self.k*self.l)+self.k*self.l*sin(theta))/(self.m*self.l*cos(theta))]
        return dydx

    def equation3(self, y0, t):
        #state = (theta, z1, phi, z2)
        # z1 = thetaDot, z2 = phiDot

        m = self.m
        l = self.l
        g = self.g

        #initial state
        theta, z1, phi, z2 = y0
        
        #definitions for first order equations
        thetaDot = z1
        phiDot = z2

        delsin = np.sin(theta - phi)
        delcos = np.cos(theta - phi)

        dydx = [thetaDot,
                (m*g*sin(phi)*delcos-m*delsin*(l*z1**2*delcos+l*z2**2)-2*m*g*sin(theta))/(l*(m+m*delsin**2)),
                phiDot,
                (2*m*(l*z1**2*delsin-g*sin(phi)+g*sin(theta)*delcos)+m*l*z2**2*delsin*delcos)/l*(m+m*delsin**2)]

        return dydx

    def solve_ODE(self):
        self.state = odeint(self.equation, self.init_state, self.time)
        x = sin(self.state[:, 0])*self.l
        y = -1*cos(self.state[:, 0])*self.l
        return (x, y)

    def solve_ODE2(self):
        self.state2 = odeint(self.equation2, self.init_state2, self.time)
        x1 = sin(self.state2[:, 0])*self.l
        y1 = -1*cos(self.state2[:, 0])*self.l
        x2 = sin(self.state2[:, 2])*self.l
        y2 = -1*cos(self.state2[:, 2])*self.l
        return (x1,y1,x2,y2)

    def solve_ODE3(self):
        self.state3 = odeint(self.equation3,self.init_state3, self.time)
        x1 = sin(self.state3[:, 0])*self.l
        y1 = -1*cos(self.state3[:, 0])*self.l
        x2 = x1 + sin(self.state3[:, 2])*self.l
        y2 = y1 + -1*cos(self.state3[:, 2])*self.l
        return (x1, y1, x2, y2)

def init():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    fig.subplots_adjust(right=.70, left=.11)
    return line, line2, line3

pend = Pendulum()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[], 'b-', lw=2)
line2, = ax.plot([],[], 'b-',lw=2)
line3, = ax.plot([],[], 'r:', marker='s', markerfacecolor='black', markersize=10.0)  

#Single Pendulum
data = pend.solve_ODE()
final = []
final.append(data[0])
final.append(data[1])
newdata = np.array(final)

#Coupled Pendulum
data2 = pend.solve_ODE2()
final2 = []
final2.append(data2[0])
final2.append(data2[1])
final2.append(data2[2])
final2.append(data2[3])
newdata2 = np.array(final2)

#DoublePendulum
data3 = pend.solve_ODE3()
final3 = []
final3.append(data3[0])
final3.append(data3[1])
final3.append(data3[2])
final3.append(data3[3])
newdata3 = np.array(final3)

#Animation functions for single, double and coupled pendulums

def animateCoupledPendulum(num,data2,line,line2, line3):
    line.set_data([-1, data2[0, num] - 1], [0, data2[1, num]])
    line2.set_data([1, data2[2, num] + 1], [0, data2[3, num]])
    line3.set_data([data2[0, num] -1, data2[2, num] + 1], [data2[1, num], data2[3, num]])
    return line, line2, line3

def animateSinglePendulum(num, data1, line, line2, line3):
    line.set_data([-1,data1[0,num]-1], [0,data1[1,num]])
    line2.set_data([1,data1[0,num]+1], [0,data1[1,num]])
    line3.set_data([data1[0,num]-1, data1[0,num]+1], [data1[1,num],data1[1,num]])
    return line, line2, line3

def animateDoublePendulum(num, data3, line, line2):
    line.set_data([0,data3[0,num]], [0,data3[1,num]])
    line2.set_data([data3[0,num],data3[2,num]], [data3[1,num],data3[3,num]])
    ax.plot(data3[2,num], data3[3,num], color='black', marker='o', ms=1.1)
    return line, line2

#ani = animation.FuncAnimation(fig, animateSinglePendulum, interval=1, frames=1000, fargs=(newdata,  line, line2, line3), init_func=init)

#ani = animation.FuncAnimation(fig, animateCoupledPendulum, interval=1, frames=5000, fargs=(newdata2,  line, line2, line3), init_func=init)

ani = animation.FuncAnimation(fig, animateDoublePendulum, interval=1, frames=600, fargs=(newdata3, line, line2), init_func=init)

plt.show()

#ani.save('dpen.mp4', fps=30, dpi=150)












