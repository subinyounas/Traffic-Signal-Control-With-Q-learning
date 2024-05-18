from gymnasium import gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class TrafficSignalEnv(gym.Env):

    def __init__(self):
        super(TrafficSignalEnv, self).__init__()

        self.max_vehicles = 40  # Constraint on maximum number of vehicles per lane
        self.cycleLength = 30  # Episode length in seconds
        self.defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}  # default green signal timers
        self.elapsedSeconds = 0  # number of elapsed seconds into the simulation
        self.signals = []  # Signals array
        self.noOfSignals = 4  # Number of signals in the simulation
        self.currentGreen = [0, 2]  # Current signals that are green
        self.nextGreen = [
            (cg + 1) % self.noOfSignals for cg in self.currentGreen
        ]  # Indicates which signal will turn green next
        self.currentYellow = 0  # Whether the current signal is displaying yellow or not

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=self.max_vehicles, shape=(2,), dtype=np.int32
        )

        # Define action space
        self.action_space = spaces.Discrete(
            29
        )  # increase/decrease duration of NS green light

        self.speeds = {
            "car": 2.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }  # average speeds of vehicles

        # Coordinates of vehicles' start
        self.x = {
            "right": [0, 0, 0],
            "down": [755, 727, 697],
            "left": [1400, 1400, 1400],
            "up": [602, 627, 657],
        }
        self.y = {
            "right": [348, 370, 398],
            "down": [0, 0, 0],
            "left": [498, 466, 436],
            "up": [800, 800, 800],
        }

        self.vehicles = {
            "right": {0: [], 1: [], 2: [], "crossed": 0},
            "down": {0: [], 1: [], 2: [], "crossed": 0},
            "left": {0: [], 1: [], 2: [], "crossed": 0},
            "up": {0: [], 1: [], 2: [], "crossed": 0},
        }
        self.state = np.array([0, 0])
        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
        self.directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

        # Coordinates of signal image, timer, and vehicle count
        self.signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
        self.signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

        # Coordinates of stop lines
        self.stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
        self.defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}
        # stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

        # Gap between vehicles
        self.stoppingGap = 15  # stopping gap
        self.movingGap = 15  # moving gap

        pygame.init()
        self.simulation = pygame.sprite.Group()

    def _get_state(self):
        # Number of vehicles in NS direction (up and down)
        ns_vehicles = sum(
            len(self.vehicles[direction][lane])
            for direction in ["up", "down"]
            for lane in [0, 1, 2]
        )
        # Number of vehicles in EW direction (left and right)
        ew_vehicles = sum(
            len(self.vehicles[direction][lane])
            for direction in ["left", "right"]
            for lane in [0, 1, 2]
        )
        return np.array([ns_vehicles, ew_vehicles])

    def step(self, action):
        green_time_ns = action + 1  # Action space is 1 to 29
        green_time_ew = self.cycleLength - green_time_ns

        self.signals[0].green = green_time_ns
        self.signals[1].green = green_time_ew
        self.signals[2].green = green_time_ns
        self.signals[3].green = green_time_ew

        self._simulate_traffic(green_time_ns, green_time_ew)

        self.state = self._get_state()

        reward = self._calculate_reward()  # Calculate the reward

        terminated = self.elapsedSeconds >= self.cycleLength
        truncated = self.elapsedSeconds >= self.cycleLength

        return self.state, reward, terminated, truncated, {}

    def reset(self):
        # Reset simulation to initial state
        self.elapsedSeconds = 0
        self.stopSim = False
        self.vehicles = {
            "right": {0: [], 1: [], 2: [], "crossed": 0},
            "down": {0: [], 1: [], 2: [], "crossed": 0},
            "left": {0: [], 1: [], 2: [], "crossed": 0},
            "up": {0: [], 1: [], 2: [], "crossed": 0},
        }
        # Reinitialize the signals
        self.signals = []
        self.state = np.array([0, 0])
        return self.state

    def close(self):
        # Optional: clean up resources
        pass

    def simulate_traffic(self, green_time_ns, green_time_ew):
        # Simulate
        self.elapsedSeconds += green_time_ns + green_time_ew

        for _ in range(green_time_ns):
            for direction in ["up", "down"]:
                for lane in [0, 1, 2]:
                    for vehicle in self.vehicles[direction][lane]:
                        vehicle.move()

        for _ in range(green_time_ew):
            for direction in ["left", "right"]:
                for lane in [0, 1, 2]:
                    for vehicle in self.vehicles[direction][lane]:
                        vehicle.move()

    # Calculate Reward
    def _calculate_reward(self):
        total_waiting_time = 0

        for direction in ["right", "down", "left", "up"]:
            for lane in [0, 1, 2]:
                for vehicle in self.vehicles[direction][lane]:
                    total_waiting_time += vehicle.waiting_time

        return -total_waiting_time

    # render
    def render(self, mode="human"):
        screen = pygame.display.set_mode((1400, 800))
        background = pygame.image.load("images/intersection.png")
        screen.blit(background, (0, 0))

        for direction in ["right", "down", "left", "up"]:
            for lane in [0, 1, 2]:
                for vehicle in self.vehicles[direction][lane]:
                    screen.blit(vehicle.image, [vehicle.x, vehicle.y])
                    vehicle.render(screen)

        pygame.display.update()

    # add a random vehicle
    def add_vehicle(vehicles, direction, lane):
        vehicle_class = random.choice(
            ["car", "bus", "truck"]
        )  # Example vehicle classes
        new_vehicle = Vehicle(
            lane, vehicle_class, direction_number=0, direction=direction
        )
        vehicles[direction][lane].append(new_vehicle)

    def close(self):
        pygame.quit()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.current = "red"
        self.signalText = ""

    def update(self, elapsed_time):
        self.timer += elapsed_time
        if self.current == "green":
            if self.timer >= self.green:
                self.current = "yellow"
                self.timer = 0
        elif self.current == "yellow":
            if self.timer >= self.yellow:
                self.current = "red"
                self.timer = 0
        elif self.current == "red":
            if self.timer >= self.red:
                self.current = "green"
                self.timer = 0


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = self.speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):  # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if direction == "right":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().width
                    - stoppingGap
                )  # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif direction == "left":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().width
                    + stoppingGap
                )
            elif direction == "down":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().height
                    - stoppingGap
                )
            elif direction == "up":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().height
                    + stoppingGap
                )
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == "right":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.direction == "right":
            if (
                self.crossed == 0
                and self.x + self.image.get_rect().width > stopLines[self.direction]
            ):  # if the image has crossed stop line now
                self.crossed = 1
            if (
                self.x + self.image.get_rect().width <= self.stop
                or self.crossed == 1
                or (any(cg in [0, 2] for cg in currentGreen) and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x + self.image.get_rect().width
                < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)
            ):
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                self.x += self.speed  # move the vehicle
        elif self.direction == "down":
            if (
                self.crossed == 0
                and self.y + self.image.get_rect().height > stopLines[self.direction]
            ):
                self.crossed = 1
            if (
                self.y + self.image.get_rect().height <= self.stop
                or self.crossed == 1
                or (any(cg in [1, 3] for cg in currentGreen) and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y + self.image.get_rect().height
                < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)
            ):
                self.y += self.speed
        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            if (
                self.x >= self.stop
                or self.crossed == 1
                or (any(cg in [0, 2] for cg in currentGreen) and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x
                > (
                    vehicles[self.direction][self.lane][self.index - 1].x
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .width
                    + movingGap
                )
            ):
                self.x -= self.speed
        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            if (
                self.y >= self.stop
                or self.crossed == 1
                or (any(cg in [1, 3] for cg in currentGreen) and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y
                > (
                    vehicles[self.direction][self.lane][self.index - 1].y
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .height
                    + movingGap
                )
            ):
                self.y -= self.speed
