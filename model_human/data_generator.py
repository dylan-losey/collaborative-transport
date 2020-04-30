import pygame
import sys
import os
import math
import numpy as np
import time
import random
import pickle
import copy


m = 1.0
b = 1.0
I = 10.0
L = 1.0
d = 50.0
rad = math.pi/180


worldx = 800
worldy = 800



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

    def human(self, follow):

        # select human hyperparameters
        mu = 0.0
        sigma = 1.0
        follow = follow

        # get the input towards goal
        direction = self.goal - np.array([self.x, self.y])
        self.dist2goal = np.linalg.norm(direction)
        magnitude = 1.0
        if self.dist2goal > 50:
            magnitude *= 50 / self.dist2goal
        fgoal = magnitude * direction

        # choose robot force
        f2 = fgoal * (1.0 - follow) + np.asarray(self.f1) * follow
        force_magnitude = np.linalg.norm(f2)
        f2 *= 50.0 / force_magnitude
        f2 += np.random.normal(mu, sigma, 2)
        self.f2 = [f2[0], f2[1]]

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

    fps = 4
    delta_t = 1.0 / fps

    r_obs = 300.0
    n_rounds = 20
    Follow = [0.0, 0.25, 0.5, 0.75, 1.0]

    for round in range(n_rounds):

        # pick goal position
        goal_angle = np.random.uniform(-math.pi/4, math.pi/4)
        goal_radius = 350
        goal_position = np.array([0.5*worldx, 0.5*worldy])
        goal_position += goal_radius * np.array([math.cos(goal_angle), math.sin(goal_angle)])

        # pick obstacle position
        obs_angle = goal_angle + np.random.uniform(-math.pi/6, math.pi/6)
        obs_radius = 100
        obs_position = np.array([0.5*worldx, 0.5*worldy])
        obs_position += obs_radius * np.array([math.cos(obs_angle), math.sin(obs_angle)])

        for follow in Follow:

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
            savename = "trajectories/r_obs_300/f_" + str(follow) + "_r_" + str(round) + ".pkl"


            while True:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                        main = False
                    if event.type == pygame.KEYUP:
                        if event.key == ord('q'):
                                pygame.quit()
                                sys.exit()
                                main = False

                table.robot(r_obs)
                table.human(follow)
                table.update(delta_t)

                data.append([table.x, table.y] + list(goal_position) + list(obs_position) + table.f1 + table.f2)
                pickle.dump(data, open(savename, "wb" ))

                if table.dist2goal < 50:
                    print(savename)
                    print("timesteps: ", len(data))
                    pygame.quit()
                    break

                world.fill((0,0,0))
                sprite_list.draw(world)

                pygame.display.flip()
                # clock.tick(fps)


if __name__ == "__main__":
    main()
