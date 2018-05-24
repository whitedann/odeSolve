import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy import sin, cos, sqrt
import matplotlib.animation as animation

class Pendulum():

    def __init__(self):
        self.l = 1.0 
        self.g = 9.8
        self.k = 0.5 
        self.m = 1.0
        self.init_state = [np.radians(45.0), 0]
        self.init_state2 = [np.radians(-45.0), 0, np.radians(0.0), 0]
        self.time = np.arange(0, 50.0, 0.01)


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



pend = Pendulum()

def init():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    return line, line2

data = pend.solve_ODE()
final = []
final.append(data[0])
final.append(data[1])
newdata = np.array(final)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(right=.70, left=.11)
line, = ax.plot([],[], 'o-')
line2, = ax.plot([],[], 'o-')
line3, = ax.plot([],[], 'o-')

###
data2 = pend.solve_ODE2()
final2 = []
final2.append(data2[0])
final2.append(data2[1])
final2.append(data2[2])
final2.append(data2[3])
newdata2 = np.array(final2)


def animate2(num,data2,line,line2, line3):
    line.set_data([-1, data2[0, num] - 1], [0, data2[1, num]])
    line2.set_data([1, data2[2, num] + 1], [0, data2[3, num]])
    line3.set_data([data2[0, num] -1, data2[2, num] + 1], [data2[1, num], data2[3, num]])
    return line, line2, line3
###


def animate(num, data1, line, line2, line3):
    line.set_data([-1,data1[0,num]-1], [0,data1[1,num]])
    line2.set_data([1,data1[0,num]+1], [0,data1[1,num]])
    line3.set_data([data1[0,num]-1, data1[0,num]+1], [data1[1,num],data1[1,num]])
    return line, line2, line3

#ani = animation.FuncAnimation(fig, animate, interval=1, frames=1000, fargs=(newdata,  line, line2, line3), init_func=init)

ani = animation.FuncAnimation(fig, animate2, interval=1, frames=5000, fargs=(newdata2,  line, line2, line3), init_func=init)
plt.show()












