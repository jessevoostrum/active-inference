import shapely.geometry as sg
import shapely as sh
import matplotlib.animation as animation
import shapely.affinity as sa
import AuxiliaryFunctions as af
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, ion, show
import Measures as me
import matplotlib



colorGoal = '#E6C200'
colorRe = '#93091F'
colorBlu = '#003C97'
ForestGreen = '#228b22'
colorT = '#2DA0A1'

matplotlib.use('TkAgg')
tau = 2*np.pi

# define the environment
arena = sg.box(-10,-10,10,10)
hole = sg.box(-7,-7,-5,7)
hole = hole.union(sg.box(-7,5,6,7))
hole = hole.union(sg.box(-7,-5,6,-7))
hole = hole.union(sg.box(0,-1,12,1))
hole = hole.union(sg.box(10,-11,11,11))
hole = hole.union(sg.box(-10,-11,-11,11))
hole = hole.union(sg.box(-11,-10,11,-11))
hole = hole.union(sg.box(11,10,-11, 11))
arena = arena.difference(hole)

walls = arena.boundary
iters = 0

bodySize = 0.35 #0.3

fig = plt.figure()
plt.rcParams["figure.figsize"] = (12, 12)
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(224)
#ax3 = fig.add_subplot(224)


def rotate_origin(obj, theta):
    return sa.rotate(obj,theta,use_radians=True,origin=(0,0))

# for plotting only
bodyPoly = sg.Point(0,0).buffer(bodySize, resolution=2)
bodyPoly = bodyPoly.union(sg.box(-0.01,-1,0.01,0))

class AgentBody:
    def __init__(self, x=0.0, y=0.0, theta=0.0, sensor_length = 1.25, env_sens_legth = 1.25):
        self.pos = sg.Point(x, y)
        self.theta = theta
        self.sens_length = sensor_length
        self.env_sens_length = env_sens_legth

    def randomValidPosition(self):
        while True:
            x, y = np.random.random(2) * 24 - 12
            self.pos = sg.Point(x, y)
            self.theta = np.random.random() * tau
            if self.touchingWall():
                return self

    def touchingWall(self):
        return arena.contains(self.pos) and (self.pos.distance(walls) > bodySize)

    def sensorValues(self, action, plot=True, plot_sensors=True):
        print("ACTION", action)
        sensors = [
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.sens_length)]), -0.1 * tau),
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.sens_length)]), 0.1 * tau),
        ]
        mySensors = [
            sa.translate(rotate_origin(s, self.theta), self.pos.x, self.pos.y)
            for s in sensors
        ]
        result = np.zeros(len(mySensors) + 1)
        for i in range(len(mySensors)):
            if hole.intersects(mySensors[i]):
                result[i] = 1
        if plot == True:
            if plot_sensors:
                for i in range(len(mySensors)):
                    if result[i]:
                        col = '#00ff00'
                    else:
                        col = '#000000'
                    plot_line(ax, mySensors[i], col)
            if plot:
                #if not self.touchingWall():
                if action == 0:
                    col = '#6699CC'
                if action == 1:
                    col = '#00FF00'
                if action == 2:
                    col = '#ffff00'
                if action == 3:
                    col = '#ff0000'
                # else:
                #     if action == 0:
                #         col = '#0000ff'
                #     if action == 1:
                #         col = '#000066'
                #     if action == 2:
                #         col = '#00FF00'
                #     if action == 3:
                #         col = '#8E7CC3'
                body = sa.translate(rotate_origin(bodyPoly, self.theta), self.pos.x, self.pos.y)
                plot_poly_fill(body, col)
        result[len(mySensors)] = self.touchingWall()
        return result

    def internal_environmental_states(self, plot =True, dt=1.0):
        env_sensors = [
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.env_sens_length)]), 0.20 * math.pi),
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.env_sens_length)]), -0.20 * math.pi),
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.env_sens_length)]), -0.80 * math.pi),
            rotate_origin(sg.LineString([(0, bodySize), (0, bodySize + self.env_sens_length)]), 0.80 * math.pi),
        ]
        myEnvSensors = [
            sa.translate(rotate_origin(s, self.theta), self.pos.x, self.pos.y)
            for s in env_sensors
        ]
        result = np.zeros(len(myEnvSensors) + 1)
        for i in range(len(myEnvSensors)):
            if hole.intersects(myEnvSensors[i]):
                result[i] = 1
        if plot:
            for i in range(len(myEnvSensors)):
                if result[i]:
                    col = '#00ff00'
                else:
                    col = '#000000'
                plot_line(ax, myEnvSensors[i], col)
        result[len(env_sensors)] = self.touchingWall()
        return result

    def update(self, actions, dt=1.0):

        turnLeft, turnRight = actions
        speed = (np.sum(actions) + 1) * 0.2
        turning = 0.04 * tau * (turnRight - turnLeft)
        self.theta += turning
        self.pos = sa.translate(self.pos,
                                -speed * np.sin(self.theta) * dt, speed * np.cos(self.theta))

    # here the agent is stuck at a wall
    def stick(self, controllerValues, dt=1.0):
        action = 0
        turnLeft, turnRight = controllerValues
        if turnLeft == 0 and turnRight == 0:
            action = 0
        if turnLeft == 1 and turnRight == 0:
            action = 1
        if turnLeft == 0 and turnRight == 1:
            action = 2
        if turnLeft == 1 and turnRight == 1:
            action = 3

        left, right, alive = self.sensorValues(action=action)
        if left == 0 and right == 0:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning
            speed = (np.sum(controllerValues) + 1) * 0.2
            self.pos = sa.translate(self.pos, -speed * np.sin(self.theta) * dt, speed * np.cos(self.theta))
        else:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning


class Controller:
    # Inner workings of the agent, here the hidden nodes are the infered environment states
    def __init__(self, n_inputs=3, n_outputs=2, n_hidden=4):
        #  self.last_c = np.array([0,0])
        self.n = 0
        self.n_pred = 0
        self.n_sa = np.zeros(pow(2, n_inputs + n_outputs))
        self.n_s = np.zeros(pow(2, 5))
        self.n_ie = np.zeros(pow(2, 5))
        self.efe = np.array([0, 0, 0, 0])

        self.n_ie_a = np.zeros(pow(2, 8))
        self.last_s = np.array([0, 0, 1])
        self.last_a = np.array([0, 0])

        self.last_int_env = np.zeros(4)
        self.goal = np.array([0.5, 0.5])
        self.p_sa = np.zeros(2 ** (n_inputs + n_outputs)) + (1 / 2 ** (n_inputs + n_outputs))

        # B matrix (5 binary internal environmental variables for t and t+1 and 2 action )
        self.gen_modelB = np.ones((pow(2, 5), pow(2, 5), pow(2, 2))) / pow(2, 5) # B.B[0] #
        # print(B.B.shape)
        # A matrix
        self.likelihoodA = np.ones((pow(2, 3), pow(2, 5))) / pow(2, 3) # A.A[0] #

        self.p_a = af.rand_cond_distr(2, n_hidden + n_inputs)
        self.p_s = np.ones(pow(2, n_inputs + n_inputs + n_outputs)) / pow(2, n_inputs)
        self.p_s_pred = np.ones(pow(2, n_inputs + n_hidden + n_outputs)) / pow(2, n_inputs)
    def getA(self):
        return self.likelihoodA
    def getB(self):
        return self.gen_modelB
    # calculate the next step and update the sampled distributions
    def sampling(self, inputValues, internal_environmental_states):
        # update the sampled probability for reaching the goal
        self.n = self.n + 1
        if inputValues[2] == 0:
            self.goal[0] = self.goal[0] * (self.n / (self.n + 1)) + (1 / (self.n + 1))
            self.goal[1] = self.goal[1] * (self.n / (self.n + 1))
        else:
            self.goal[1] = self.goal[1] * (self.n / (self.n + 1)) + (1 / (self.n + 1))
            self.goal[0] = self.goal[0] * (self.n / (self.n + 1))

        index_sa = int(af.getIndex(self.last_s, self.last_a, []))
        for i in range(len(self.p_sa)):
            self.p_sa[i] = self.p_sa[i] * (self.n / (self.n + 1))
        self.p_sa[index_sa] = self.p_sa[index_sa] + (1 / (self.n + 1))

        index_ssa = int(af.getIndex(self.last_s, self.last_a, inputValues))
        print(self.last_s, self.last_a, inputValues, index_ssa)

        index_sa = int(af.getIndex(self.last_s, self.last_a, []))
        self.n_sa[index_sa] = self.n_sa[index_sa] + 1
        for i in range(len(self.p_s)):
            if int((i // (pow(2, len(inputValues))))) == index_sa:
                if i == index_ssa:
                    self.p_s[i] = ((self.n_sa[index_sa]) / (self.n_sa[index_sa] + 1)) * self.p_s[i] + (
                            1 / (self.n_sa[index_sa] + 1))
                else:
                    self.p_s[i] = ((self.n_sa[index_sa]) / (self.n_sa[index_sa] + 1)) * self.p_s[i]
            else:
                self.p_s[i] = self.p_s[i]
        # sample the generative model
        index_ie_a = int(af.getIndex(self.last_int_env, self.last_a, []))
        index_ie_ie_a = int(af.getIndex(self.last_int_env, self.last_a, internal_environmental_states))
        # for the matrix representation
        index_iet = int(af.getIndex(internal_environmental_states, [], []))
        index_ie_past = int(af.getIndex(self.last_int_env, [], []))
        index_a = int(af.getIndex(self.last_a, [], []))
        self.n_ie_a[index_ie_a] = self.n_ie_a[index_ie_a] + 1

        for i in range(pow(2, 5)):
            for j in range(pow(2, 5)):
                for k in range(pow(2, 2)):
                    if index_ie_past == j and index_a == k:
                        if index_iet == i:
                            self.gen_modelB[i, j, k] = ((self.n_ie_a[index_ie_a]) / (self.n_ie_a[index_ie_a] + 1)) * \
                                                       self.gen_modelB[i, j, k] + (1 / (self.n_ie_a[index_ie_a] + 1))
                        else:
                            self.gen_modelB[i, j, k] = ((self.n_ie_a[index_ie_a]) / (self.n_ie_a[index_ie_a] + 1)) * \
                                                       self.gen_modelB[i, j, k]
                    else:
                        self.gen_modelB[i, j, k] = self.gen_modelB[i, j, k]

#        print(self.gen_model[0], self.gen_modelB[0, 0, 0])

        # sample the likelihood
        index_ie = int(af.getIndex(internal_environmental_states, [], []))
        index_ie_s = int(af.getIndex(internal_environmental_states, inputValues, []))
        index_s = int(af.getIndex(inputValues, [], []))
        self.n_ie[index_ie] = self.n_ie[index_ie] + 1

        for i in range(pow(2, 3)):
            for j in range(pow(2, 5)):
                if index_iet == j:
                    if index_s == i:
                        self.likelihoodA[i, j] = ((self.n_ie[index_ie]) / (self.n_ie[index_ie] + 1)) * self.likelihoodA[
                            i, j] + (
                                                         1 / (self.n_ie[index_ie] + 1))
                    else:
                        self.likelihoodA[i, j] = ((self.n_ie[index_ie]) / (self.n_ie[index_ie] + 1)) * self.likelihoodA[
                            i, j]
                else:
                    self.likelihoodA[i, j] = self.likelihoodA[i, j]
        return  0 #outputValues



class Agent:
    def __init__(self, it, sensor_length = 1.25, env_sensor_length = 1.25,  x=0.0, y=0.0, theta=0.0):
        self.body = AgentBody(x, y, theta, sensor_length, env_sensor_length)
        self.controller = Controller()
        self.it = it
        self.c = [0,0,0,0]


    def alive(self):
        return self.body.touchingWall()

    def set(self, it):
        self.it = it

    def reset(self, it):
        self.body.randomValidPosition()
        self.it = it
        return self
    def step_animated(self, actions, dt=1.0,sampling =False, plot=True, plot_sensors=True):
        ani = animation.FuncAnimation(fig, self.step(actions))
        plt.show()
      #  return sensorValues, internal_environmental_states
    def step(self, actions, iteration, dt=1.0,sampling =False, plot=True, trajectorie=False, plot_sensors=True):
        if self.controller.last_a[1] == 0 and self.controller.last_a[0] == 0:
            action = 0
        if self.controller.last_a[1] == 0 and self.controller.last_a[0] == 1:
            action = 1
        if self.controller.last_a[1] == 1 and self.controller.last_a[0] == 0:
            action = 2
        if self.controller.last_a[1] == 1 and self.controller.last_a[0] == 1:
            action = 3
        sensorValues = self.body.sensorValues(plot=plot, action=action, plot_sensors=plot_sensors, )
        internal_environmental_states = self.body.internal_environmental_states()
        self.controller.last_s = sensorValues
        self.controller.last_a = actions
        self.controller.last_int_env = internal_environmental_states
        print("in step", self.controller.last_s, self.controller.last_a, self.controller.last_int_env)
        if sampling:
            self.controller.sampling(sensorValues,internal_environmental_states)
        if sensorValues[2]:
            self.body.update(actions)
        else:
            self.body.stick(actions)
        if plot:
            plot_arena()
            if iteration%100==0:
                print(iteration)
                d = me.calc_meas_morph(self.controller.p_sa, self.controller.p_s)
                print("Morph", d)
                d = np.append(d, self.controller.goal[1])
                for i in range(len(self.c)):
                    self.c[i] = np.append(self.c[i], d[i])
                ax1.clear()
              #  ax3.clear()
                ax2.clear()
                plotmeasures(self.c, iteration // 100)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.show()
            if not trajectorie:
                ax.clear()

        return sensorValues, internal_environmental_states


from descartes.patch import PolygonPatch
COLOR = {
        True:  '#6699cc',
        False: '#ff3333'
        }
def v_color(ob):
        return COLOR[ob.is_valid]
def plot_poly(polygon, col=None):
        if not col:
            col = v_color(polygon)
    #	patch = PolygonPatch(polygon, facecolor=col, edgecolor=col, alpha=1, zorder=2)
        patch = PolygonPatch(polygon, fill=False, edgecolor=col, alpha=1, zorder=2)
        ax.add_patch(patch)


def plot_poly_fill(polygon, col=None):
    if not col:
        col = v_color(polygon)
    #	patch = PolygonPatch(polygon, facecolor=col, edgecolor=col, alpha=1, zorder=2)
    patch = PolygonPatch(polygon, fill=True, edgecolor=col, color = col, alpha=1, zorder=2)
    ax.add_patch(patch)
def plot_line(ax, ob, col='#000000'):
        x, y = ob.xy
        ax.plot(x, y, color=col, alpha=1, linewidth=1, solid_capstyle='round', zorder=2)


def plot_arena():
    plot_poly(arena)
   # #print(walls.bounds)
    x_range = walls.bounds[0]-2,walls.bounds[2]+2
    y_range = walls.bounds[1]-2,walls.bounds[3]+2
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect(1)


def plotmeasures(c, n):
        print(n, np.arange(n+2), c[1])
        ax1.plot(np.arange(n+2), c[3], color='yellow', label="Goal")
        ax2.plot(np.arange(n+2), c[0], color='green', label="Morphological Computation")
      #  ax3.plot(np.arange(n+2), c[1], color='blue', label="Action Effect")
      #  ax3.plot(np.arange(n+2), c[2], color='red', label="Synergistic")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
      #  ax3.legend(loc="upper left")




