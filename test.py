import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy import sin, cos
import matplotlib.animation as animation

class Pendulum():

    def __init__(self):
        self.l = 2.0
        self.g = 9.8
        self.init_state = [np.radians(10.0), np.radians(0.0)]
        self.time = np.arange(0, 30.0, 0.025)


    def equation(self, y0,t):
        theta, x = y0
        f = [x, -(self.g/self.l)*sin(theta)]
        return f

    def solve_ODE(self):
        self.state = odeint(self.equation, self.init_state, self.time)
        x = sin(self.state[:, 0])*self.l
        y = -1*cos(self.state[:, 0])*self.l
        return (x, y)

    def get_time(self):
        return self.time


pend = Pendulum()

def init():
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    line.set_data([], [])
    return line,

data = pend.solve_ODE()
xdata = []
ydata= []
xdata.append(data[0])
ydata.append(data[1])

final = []
final.append(xdata)
final.append(ydata)

newdata = np.array(final)

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[], 'b-')

print(newdata)


def animate(num, datas,line):
    line.set_data([0,datas[0,0,num]], [1,datas[1,0,num]])
    return line,

ani = animation.FuncAnimation(fig, animate, interval=10, frames=1000, fargs=(newdata,line), blit=True, init_func=init)

plt.show()












