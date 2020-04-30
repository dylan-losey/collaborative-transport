import pygame
import sys
import os
import math
import numpy as np
import time
import random
import pickle
import copy
from model import LSTM_LSTM
import torch


m = 1.0
b = 1.0
I = 10.0
L = 1.0
d = 50.0
rad = math.pi/180


worldx = 800
worldy = 800



class Model(object):

    def __init__(self, modelname):
        self.model = LSTM_LSTM()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def encode(self, x):
        return self.model.encode(x)

    def init_hidden(self):
        return self.model.init_hidden()

    def step(self, s, z, hidden):
        return self.model.step(s, z, hidden)


def net_display(screen, f, x, y, thickness=5, trirad=8):

    start = [x, y]
    fx = f[0]
    fy = f[1]
    end = [0,0]
    end[0] = start[0]+fx
    end[1] = start[1]+fy

    lcolor = (255, 179, 128)
    tricolor = (255, 179, 128)
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
    pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                        end[1] + trirad * math.cos(rotation)),
                                       (end[0] + trirad * math.sin(rotation - 120*rad),
                                        end[1] + trirad * math.cos(rotation - 120*rad)),
                                       (end[0] + trirad * math.sin(rotation + 120*rad),
                                        end[1] + trirad * math.cos(rotation + 120*rad))))


class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self):
        pygame.event.get()
        fx = self.gamepad.get_axis(0)
        fy = self.gamepad.get_axis(1)
        e_stop = self.gamepad.get_button(7)
        f = np.asarray([fx * 50.0, fy * 50.0])
        if np.linalg.norm(f) > 50.0:
            f *= 50.0 / np.linalg.norm(f)
        return list(f), e_stop


class Obstacle(pygame.sprite.Sprite):

    def __init__(self, position):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.original_img = pygame.image.load(os.path.join('images','obstacle.png')).convert()
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.rect  = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2


class Target(pygame.sprite.Sprite):

    def __init__(self, position):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.original_img = pygame.image.load(os.path.join('images','target.png')).convert()
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.rect  = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2


class Table(pygame.sprite.Sprite):

    def __init__(self, target_position, obstacle_position):

        # create goal
        self.goal = target_position
        self.dist2goal = None

        # create obstacle
        self.obs = obstacle_position
        self.dist2obs = None

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.original_img = pygame.image.load(os.path.join('images','table.png')).convert()
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.rect  = self.image.get_rect()

        # initial conditions
        self.x_speed = 0.0
        self.y_speed = 0.0
        self.angle_speed = 0.0
        self.x = 0.25*worldx
        self.y = 0.5*worldy
        self.angle = 0.0
        self.rect.x = self.x
        self.rect.y = self.y
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2
        self.image = pygame.transform.rotate(self.original_img, math.degrees(self.angle))

        # current inputs
        self.f1 = [0,0]
        self.f2 = [0,0]

    def acceleration(self, f1, f2):

        # equations of motion for table
        a_x = -b/m * self.x_speed + 1/m * (f1[0] + f2[0])
        a_y = -b/m * self.y_speed + 1/m * (f1[1] + f2[1])
        M_z = L / 2.0 * (math.sin(self.angle)*(f1[0] - f2[0]) - math.cos(self.angle)*(-f1[1] + f2[1]))
        a_angle = -d/I * self.angle_speed + 1/I * M_z
        return a_x, a_y, a_angle

    def robot(self, r_obs):

        # select robot hyperparameters
        mu = 0.0
        sigma = 1.0
        r_obs = r_obs

        # get the input towards goal
        direction = self.goal - np.array([self.x, self.y])
        self.dist2goal = np.linalg.norm(direction)
        magnitude = 1.0
        if self.dist2goal > 50:
            magnitude *= 50 / self.dist2goal
        fgoal = magnitude * direction

        # get input away from obstacle
        avoid = self.obs - np.array([self.x, self.y])
        self.dist2obs = r_obs - np.linalg.norm(avoid)
        fobs = 0.0 * avoid
        if self.dist2obs > 0:
            fobs = - self.dist2obs * (avoid / np.linalg.norm(avoid))

        # choose robot force
        f1 = fgoal + fobs
        force_magnitude = np.linalg.norm(f1)
        f1 *= 50.0 / force_magnitude
        f1 += np.random.normal(mu, sigma, 2)
        self.f1 = [f1[0], f1[1]]

    def update(self, delta_t):

        # convert inputs to the table acceleration
        a_x, a_y, a_angle = self.acceleration(self.f1, self.f2)

        # integrate to get velocity
        self.x_speed += a_x * delta_t
        self.y_speed += a_y * delta_t
        self.angle_speed += a_angle * delta_t

        # integrate to get position
        self.x += self.x_speed * delta_t
        self.y += self.y_speed * delta_t
        self.angle += self.angle_speed * delta_t

        # update the table position
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2
        self.image = pygame.transform.rotate(self.original_img, math.degrees(self.angle))


def main():

    # you choose latent z in [0, 1]
    z = (float(sys.argv[1]) - 1.0) * 3

    # you choose the robot role in [0, 1]
    rr = float(sys.argv[2])

    fps = 4
    delta_t = 1.0 / fps

    r_obs = 100.0 * rr + 200.0
    modelname = 'models/lstm_model.pt'
    model = Model(modelname)

    # pick goal position
    goal_angle = np.random.uniform(-math.pi/6, math.pi/6)
    goal_radius = 350
    goal_position = np.array([0.5*worldx, 0.5*worldy])
    goal_position += goal_radius * np.array([math.cos(goal_angle), math.sin(goal_angle)])

    # pick obstacle position
    obs_angle = goal_angle + np.random.uniform(math.pi/6, math.pi/4)
    obs_radius = 100
    obs_position = np.array([0.5*worldx, 0.5*worldy])
    obs_position += obs_radius * np.array([math.cos(obs_angle), math.sin(obs_angle)])

    pygame.init()
    clock = pygame.time.Clock()
    world = pygame.display.set_mode([worldx,worldy])

    table = Table(goal_position, obs_position)
    target = Target(goal_position)
    obs = Obstacle(obs_position)
    sprite_list = pygame.sprite.Group()
    sprite_list.add(target)
    sprite_list.add(obs)
    sprite_list.add(table)

    data = []
    zt = torch.tensor(z).view(1)
    hidden = model.init_hidden()

    while len(data) < 25:

        table.robot(r_obs)

        table_pos = [(table.x-400)/350.0, (table.y-400)/350.0]
        goal_pos = [(goal_position[0]-400)/350.0, (goal_position[1]-400)/350.0]
        partner_force = [table.f1[0]/50.0, table.f1[1]/50.0]
        s = torch.tensor(table_pos + goal_pos + partner_force)
        ahat, hidden = model.step(s, zt, hidden)
        ahat = ahat.detach().numpy()
        table.f2 = list(ahat * 50)

        table.update(delta_t)

        data.append([table.x, table.y] + list(goal_position) + table.f1 + table.f2)

        world.fill((0,0,0))
        sprite_list.draw(world)
        net_display(world, table.f1, table.x, table.y)

        pygame.display.flip()
        clock.tick(fps)


    data_normalized = []
    for item in data:
        x = [0] * 8
        x[0] = (item[0] - 400) / 350.0
        x[1] = (item[1] - 400) / 350.0
        x[2] = (item[2] - 400) / 350.0
        x[3] = (item[3] - 400) / 350.0
        x[4] = item[4] / 50.0
        x[5] = item[5] / 50.0
        x[6] = item[6] / 50.0
        x[7] = item[7] / 50.0
        data_normalized.append(x)
    traj = torch.tensor(data_normalized)
    z_pred = model.encode(traj).item()
    print("recovered latent variable:", z_pred/3 + 1)


if __name__ == "__main__":
    main()
