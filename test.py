import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy import sin, cos, sqrt
import matplotlib.animation as animation

class Pendulum():

    def __init__(self):
        self.l = 1.5
        self.g = 9.8
        self.init_state = [np.radians(45.0), 0]
        self.time = np.arange(0, 30.0, 0.01)


    def equation(self, y0,t):
        theta, x = y0
        f = [x, -(self.g/self.l)*sin(theta)]
        return f

    def equation2(self, y0, t):
        theta, x = y0
        f = [x, -(self.g/self.l)*theta]
        return f

    def solve_ODE(self):
        self.state = odeint(self.equation, self.init_state, self.time)
        x = sin(self.state[:, 0])*self.l
        y = -1*cos(self.state[:, 0])*self.l
        return (x, y)

    def solve_ODE2(self):
        self.state2 = odeint(self.equation2, self.init_state, self.time)
        x = sin(self.state2[:, 0])*self.l
        y = -1*cos(self.state2[:, 0])*self.l
        return (x, y)



pend = Pendulum()

def init():
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    return line, line2

data = pend.solve_ODE()
data2 = pend.solve_ODE2()

for i in range (0,100):
    print((data[0][i]**2 + data[1][i]**2)**(1/2), (data2[0][i]**2 + data2[1][i]**2)**(1/2))

final = []
final.append(data[0])
final.append(data[1])
final2 = []
final2.append(data2[0])
final2.append(data2[1])

newdata = np.array(final)
newdata2 = np.array(final2)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(right=.70, left=.11)
line, = ax.plot([],[], 'o-')
line2, = ax.plot([],[], 'o-')
line3, = ax.plot([],[], 'o-')


def animate(num, data1, data2, line, line2, line3):
    line.set_data([-1,data1[0,num]-1], [0,data1[1,num]])
    line2.set_data([1,data1[0,num]+1], [0,data1[1,num]])
    line3.set_data([data1[0,num]-1, data1[0,num]+1], [data1[1,num],data1[1,num]])
    return line, line2, line3

ani = animation.FuncAnimation(fig, animate, interval=1, frames=1000, fargs=(newdata, newdata2, line, line2, line3), blit=True, init_func=init)

plt.show()












