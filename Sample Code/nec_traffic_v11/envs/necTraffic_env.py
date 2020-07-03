from abc import ABC
from gym import Env
from gym import spaces
from gym.utils import seeding
from string import Template
import numpy as np
import os
import sys
import time
import random
import traci

####################################################################################################
########################################## SET SUMO PATH ###########################################
####################################################################################################

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

errors = (traci.FatalTraCIError, traci.TraCIException, KeyError)
traci_count = traci.lanearea.getLastStepVehicleNumber
traci_jam = traci.lanearea.getJamLengthVehicle
speed = traci.lanearea.getLastStepMeanSpeed

def sigmoid(x):
    sig = 4.5 * 1 / (1 + np.exp(0.004 * (x - 1500))) - 0.5
    return sig

####################################################################################################
####################################################################################################

####################################################################################################
#################################### TRAFFIC LIGHT SETTINGS ########################################
####################################################################################################

class TrafficLight:

    def __init__(self):
        self.states = {
            0: "grrrrgGGGrgrrrrgGGGrrGrGr",
            1: "grrrrgyyyrgrrrrgyyyrrrrrr",
            2: "grrrrgrrrrgrrrrgrrrrrrrrr",

            3: "grrrrgrrrGgrrrrgrrrGGrrrr",
            4: "grrrrgrrrygrrrrgrrryyrrrr",
            5: "grrrrgrrrrgrrrrgrrrrrrrrr",

            6: "gGGGrgrrrrgGGrrgrrrrrrGrG",
            7: "gyyyrgrrrrgyyrrgrrrrrrrrr",
            8: "grrrrgrrrrgrrrrgrrrrrrrrr",

            9: "grrrGgrrrrgrrGGgrrrrrrrrr",
           10: "grrrygrrrrgrryygrrrrrrrrr",
           11: "grrrrgrrrrgrrrrgrrrrrrrrr"
        }
        self.phase = 0
        self.signal = "grrrrgGGGrgrrrrgGGGrrGrGr"

    def next_phase(self, phase):
        self.phase = phase
        self.signal = self.states[self.phase]
        return self.phase, self.signal

    def terminate_green(self):
        self.phase = np.mod(self.phase + 1, len(self.states))
        self.signal = self.states[self.phase]
        return self.phase, self.signal

####################################################################################################
####################################################################################################

####################################################################################################
######################################### SUMO ENVIRONMENT #########################################
####################################################################################################

class NECTraffic(Env, ABC):
    metadata = {'render.modes': ['human', 'rgb_array']}

    ######################################## CONFIGURATION #########################################

    def __init__(self):

        self.simulation_end = 400
        self.time_step = 0
        self.sumo_step = 0
        self.green_time = 10
        self.yellow_time = 3
        self.red_time = 2
        self.num_flicker = None
        self.restart_delay = 0
        self.np_random = None
        self.route_info = None
        self.folder = None
        self.action_taken = False
        self.action_space = spaces.Discrete(4)
        self.light = TrafficLight()

        self.det_light_n = ["detL_N_1", "detL_N_2", "detL_N_3", "detL_N_4"]              # Sensors on N
        self.det_light_e = ["detL_E_1", "detL_E_2", "detL_E_3", "detL_E_4"]              # Sensors on E
        self.det_light_s = ["detL_S_1", "detL_S_2", "detL_S_3", "detL_S_4", "detL_S_5"]  # Sensors on S
        self.det_light_w = ["detL_W_1", "detL_W_2", "detL_W_3", "detL_W_4", "detL_W_5"]  # Sensors on W

        self.det_count_n = ["detC_N_1", "detC_N_2", "detC_N_3", "detC_N_4"]              # Toral num of vehicles on N
        self.det_count_e = ["detC_E_1", "detC_E_2", "detC_E_3", "detC_E_4"]              # Total num of vehicles on E
        self.det_count_s = ["detC_S_1", "detC_S_2", "detC_S_3", "detC_S_4", "detC_S_5"]  # Total num of vehicles on S
        self.det_count_w = ["detC_W_1", "detC_W_2", "detC_W_3", "detC_W_4", "detC_W_5"]  # Total num of vehicles on W

        self.seed()

    ####################################### SUMO AND GYM TOOLKIT #######################################

    def reset(self, gui, folder, options):

        try:
            traci.close()
        except errors:
            pass

        if self.restart_delay > 0:
            time.sleep(self.restart_delay)

        basepath = os.path.join(os.path.dirname(__file__), "config")
        netfile = os.path.join(basepath, "traffic.net.xml")
        routefile = os.path.join(basepath, "traffic.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "traffic.add.xml")

        self.folder = folder

        if self.folder == "RL":
            if all(options):
                tmpfile = "Plots/" + self.folder + "/all/tmp_all.rou.xml"
            elif not any(options):
                tmpfile = "Plots/" + self.folder + "/vanilla_dqn/tmp_vanilla_dqn.rou.xml"
            else:
                options_list = ["double", "dueling", "per", "noisy", "dist"]
                zero_idx = list(options).index(0)
                tmpfile = "Plots/" + self.folder + "/no_" + options_list[zero_idx] + "_dqn/tmp_no_" + options_list[zero_idx] + "_dqn.rou.xml"
        else:
            tmpfile = "Plots/" + self.folder + "/tmp_{}".format(self.folder) + ".rou.xml"

        self.args = ["--net-file", netfile, "--route-files", tmpfile, "--additional-files", addfile]
        self.args += ["-S", "-Q", "--gui-settings-file", guifile]
        self.tmpfile = tmpfile

        with open(routefile) as f:
            self.route = f.read()

        self.write_routes()

        if gui:
            sumo_cmd = ["sumo-gui"] + self.args
            self.mode = "gui"
        else:
            sumo_cmd = ["sumo"] + self.args
            self.mode = "non-gui"

        traci.start(sumo_cmd)

        self.sumo_step = 0
        self.num_flicker = 0
        self.sumo_running = True

        self.total_num_n, self.num_n_s, self.num_n_e = 0, 0, 0
        self.total_num_e, self.num_e_w, self.num_e_s = 0, 0, 0
        self.total_num_s, self.num_s_n, self.num_s_w = 0, 0, 0
        self.total_num_w, self.num_w_e, self.num_w_n = 0, 0, 0

        self.queue_NS = 0
        self.queue_EW = 0

        state = [0] * (4 * self.green_time + 1)

        return state

    def run_simulation(self):

        self.total_num_n += sum([traci_count(self.det_count_n[i]) for i in range(4)])
        self.total_num_e += sum([traci_count(self.det_count_e[i]) for i in range(4)])
        self.total_num_s += sum([traci_count(self.det_count_s[i]) for i in range(5)])
        self.total_num_w += sum([traci_count(self.det_count_w[i]) for i in range(5)])

        self.num_n_s = sum([traci_count(self.det_light_n[i]) for i in range(3)])
        self.num_n_e = traci_count(self.det_light_n[3])

        self.num_e_w = sum([traci_count(self.det_light_e[i]) for i in range(3)])
        self.num_e_s = traci_count(self.det_light_e[3])

        self.num_s_n = sum([traci_count(self.det_light_s[i]) for i in range(3)])
        self.num_s_w = sum([traci_count(self.det_light_s[i]) for i in range(3,5)])

        self.num_w_e = sum([traci_count(self.det_light_w[i]) for i in range(3)])
        self.num_w_n = sum([traci_count(self.det_light_w[i]) for i in range(3,5)])

        if self.sumo_step < 3:
            self.queue_n, self.queue_e, self.queue_s, self.queue_w = 0, 0, 0, 0
        else:
            self.queue_n = sum([traci_jam(self.det_light_n[i]) for i in range(4)])
            self.queue_e = sum([traci_jam(self.det_light_e[i]) for i in range(4)])
            self.queue_s = sum([traci_jam(self.det_light_s[i]) for i in range(5)])
            self.queue_w = sum([traci_jam(self.det_light_w[i]) for i in range(5)])

        self.queue_NS += self.queue_n + self.queue_s
        self.queue_EW += self.queue_e + self.queue_w

        # if self.mode == "gui" and self.sumo_step % 1 == 0:
        #     traci.gui.screenshot('View #0', "Frames/" + self.folder + "/Screenshot_{}.png".format(self.sumo_step))
        #     time.sleep(0.01)

        traci.simulationStep()
        self.sumo_step += 1
        time.sleep(self.time_step)

    def route_sample(self):
        # random.seed(random.randint(0, 100000000))
        random.seed(4)

        self.prob = {
            # Traffic from east
            'ew': 0.008,
            'es': 0.005,
            'en': 0.005,

            # Traffic from north
            'nw': 0.03,
            'ne': 0.03,
            'ns': 0.2,

            # Traffic from west
            'we': 0.008,
            'wn': 0.005,
            'ws': 0.005,

            # Traffic from south
            'se': 0.03,
            'sw': 0.03,
            'sn': 0.2,

            # Pedestrian generator
            'pn': random.uniform(0.005, 0.03),
            'pe': random.uniform(0.005, 0.03),
            'ps': random.uniform(0.005, 0.03),
            'pw': random.uniform(0.005, 0.03)
        }

        # self.prob = {
        #     # Traffic from east
        #     'ew': random.uniform(0.005, 0.08),
        #     'es': random.uniform(0.005, 0.04),
        #     'en': random.uniform(0.005, 0.04),
        #
        #     # Traffic from north
        #     'nw': random.uniform(0.005, 0.02),
        #     'ne': random.uniform(0.005, 0.02),
        #     'ns': random.uniform(0.005, 0.04),
        #
        #     # Traffic from west
        #     'we': random.uniform(0.005, 0.08),
        #     'wn': random.uniform(0.005, 0.04),
        #     'ws': random.uniform(0.005, 0.04),
        #
        #     # Traffic from south
        #     'se': random.uniform(0.005, 0.02),
        #     'sw': random.uniform(0.005, 0.02),
        #     'sn': random.uniform(0.005, 0.04),
        #
        #     # Pedestrian generator
        #     'pn': random.uniform(0.005, 0.02),
        #     'pe': random.uniform(0.005, 0.02),
        #     'ps': random.uniform(0.005, 0.02),
        #     'pw': random.uniform(0.005, 0.02)
        # }

        return {
            "en": self.prob['en'], "ew": self.prob['ew'], "es": self.prob['es'],
            "nw": self.prob['nw'], "ns": self.prob['ns'], "ne": self.prob['ne'],
            "ws": self.prob['ws'], "we": self.prob['we'], "wn": self.prob['wn'],
            "se": self.prob['se'], "sn": self.prob['sn'], "sw": self.prob['sw'],
            "pn": self.prob['pn'], "pe": self.prob['pe'], "ps": self.prob['ps'],
            "pw": self.prob['pw']
        }

    def write_routes(self):
        self.route_info = self.route_sample()
        with open(self.tmpfile, 'w') as f:
            f.write(Template(self.route).substitute(self.route_info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ##################################################################################################
    ######################################### GYM ENVIRONMENT ########################################
    ##################################################################################################

    ########################################### OBSERVATION  #########################################

    def get_state(self):

        state = [
                    (self.num_n_s + self.num_s_n) / 10, (self.num_n_e + self.num_s_w) / 10,
                    (self.num_e_w + self.num_w_e) / 10, (self.num_e_s + self.num_w_n) / 10
                ]

        return state

    ############################################# REWARD ############################################

    def get_reward(self):

        reward = 0.0

        num_of_vehicles = sum(self.get_state())
        queue = self.queue_n + self.queue_e + self.queue_s + self.queue_w
        flicker = self.action_taken
        sum_of_prob = sum(self.prob.values())

        # Main reward definition
        reward -= ((1 / sum_of_prob) * queue) / 5000
        reward -= 0.01 * flicker

        # Facilitating rewards
        if num_of_vehicles == 0 and flicker:
            reward -= 0.2

        # End of episode bonus rewards
        done = self.sumo_step > self.simulation_end
        total_wait = self.queue_NS + self.queue_EW

        if done:
            reward = sigmoid(total_wait)

        return reward

    ######################################### SIDE INFORMATION ######################################

    def get_info(self):

        total_num_NS = self.total_num_n + self.total_num_s
        total_num_EW = self.total_num_e + self.total_num_w

        return [
                   self.sumo_step, self.num_flicker,
                  (self.queue_NS, self.queue_EW),
                  (total_num_NS, total_num_EW),
                  ([self.num_n_s, self.num_n_e], [self.num_e_w, self.num_e_s],
                   [self.num_s_n, self.num_s_w], [self.num_w_e, self.num_w_n])
               ]

    ########################################## STEP FUNCTION #########################################

    def step(self, action):

        Light = self.light
        phase = 3 * action
        next_state = []
        reward = 0.0

        # Similar action (traffic light remains the same)
        if phase == Light.phase:

            self.action_taken = False

            for i in range(self.green_time):
                self.run_simulation()
                # time.sleep(0.03)
                next_state.extend(self.get_state())
                if not self.sumo_step > self.simulation_end + 1:
                    reward += self.get_reward()

        # Different action (traffic light switches)
        else:
            self.num_flicker += 1
            self.action_taken = True

            Light.phase, Light.signal = Light.terminate_green()
            traci.trafficlight.setRedYellowGreenState("0", Light.signal)

            # Yellow Phase
            for i in range(self.yellow_time):
                self.run_simulation()
                # time.sleep(0.03)
                if not self.sumo_step > self.simulation_end + 1:
                    reward += self.get_reward()
                self.action_taken = False

            Light.phase, Light.signal = Light.terminate_green()
            traci.trafficlight.setRedYellowGreenState("0", Light.signal)

            # All Red Phase
            for i in range(self.red_time):
                self.run_simulation()
                # time.sleep(0.03)
                if not self.sumo_step > self.simulation_end + 1:
                    reward += self.get_reward()

            Light.phase, Light.signal = Light.next_phase(phase)
            traci.trafficlight.setRedYellowGreenState("0", Light.signal)

            # Next Green-Red Phase
            for i in range(self.green_time):
                self.run_simulation()
                # time.sleep(0.03)
                next_state.extend(self.get_state())
                if not self.sumo_step > self.simulation_end + 1:
                    reward += self.get_reward()

        info = self.get_info()
        next_state.append(Light.phase)
        done = self.sumo_step > self.simulation_end

        return next_state, reward, done, info

####################################################################################################
####################################################################################################