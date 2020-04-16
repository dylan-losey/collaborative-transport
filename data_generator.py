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


class Player(pygame.sprite.Sprite):

    def __init__(self):

        # create goal
        goal_angle = np.random.uniform(-math.pi, math.pi)
        goal_radius = np.random.uniform(300,360)
        self.goal = np.array([0.5*worldx, 0.5*worldy])
        self.goal += goal_radius * np.array([math.cos(goal_angle), math.sin(goal_angle)])
        self.dist2goal = None

        # create obstacle
        obs_angle = goal_angle + np.random.uniform(-math.pi/2, math.pi/2)
        obs_radius = np.random.uniform(150,200)
        self.obs = np.array([0.5*worldx, 0.5*worldy])
        self.obs += obs_radius * np.array([math.cos(obs_angle), math.sin(obs_angle)])
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
        self.x = 0.5*worldx
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

        # noise properties
        self.mu = 0.0
        self.sigma = 30.0

        # trust properties
        self.time2hold = 10
        self.n_steps = 0
        self.trust = None
        self.trust_param = None

    def acceleration(self, f1, f2):

        # equations of motion for table
        a_x = -b/m * self.x_speed + 1/m * (f1[0] + f2[0])
        a_y = -b/m * self.y_speed + 1/m * (f1[1] + f2[1])
        M_z = L / 2.0 * (math.sin(self.angle)*(f1[0] - f2[0]) - math.cos(self.angle)*(-f1[1] + f2[1]))
        a_angle = -d/I * self.angle_speed + 1/I * M_z
        return a_x, a_y, a_angle

    def update(self, delta_t):

        # get the table
        self.rect = self.image.get_rect(center=self.rect.center)

        # get the input towards goal
        direction = self.goal - np.array([self.x, self.y])
        self.dist2goal = np.linalg.norm(direction)
        magnitude = 1.0
        if self.dist2goal > 50:
            magnitude *= 50 / self.dist2goal
        fgoal = magnitude * direction

        # get input away from obstacle
        avoid = self.obs - np.array([self.x, self.y])
        self.dist2obs = np.linalg.norm(avoid)
        magnitude = 0.0
        if self.dist2obs < 150:
            magnitude = -(150.0 - self.dist2obs)**2
        fobs = magnitude * avoid

        # choose robot force
        f1 = fgoal + fobs
        force_magnitude = np.linalg.norm(f1)
        if force_magnitude > 50:
            f1 *= 50 / force_magnitude
        f1 += np.random.normal(self.mu, self.sigma, 2)

        # choose human force
        if not self.n_steps % self.time2hold:
            self.trust = (np.random.random() < self.trust_param)
        self.n_steps += 1
        f2 = fgoal
        if self.trust:
            f2 = copy.deepcopy(f1)
        f2 += np.random.normal(self.mu, self.sigma, 2)

        # save the inputs
        self.f1 = [f1[0], f1[1]]
        self.f2 = [f2[0], f2[1]]

        # convert these inputs to the table acceleration
        a_x, a_y, a_angle = self.acceleration(f1, f2)

        # integrate to get velocity
        self.x_speed += a_x * delta_t
        self.y_speed += a_y * delta_t
        self.angle_speed += a_angle * delta_t

        # integrate to get position
        self.x += self.x_speed * delta_t
        self.y += self.y_speed * delta_t
        self.angle += self.angle_speed * delta_t

        # update the table position
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2
        self.image = pygame.transform.rotate(self.original_img, math.degrees(self.angle))


def main():

    num_of_runs = int(sys.argv[1])
    trust = float(sys.argv[2])
    fps = 40

    for count in range(num_of_runs):

        file_name = "simulated-data/t" + str(trust) + "_r" + str(count) + ".pkl"

        clock = pygame.time.Clock()
        pygame.init()
        world = pygame.display.set_mode([worldx,worldy])

        player = Player()
        target = Target(player.goal)
        obs = Obstacle(player.obs)
        sprite_list = pygame.sprite.Group()
        sprite_list.add(target)
        sprite_list.add(obs)
        sprite_list.add(player)

        data = []
        prev_time = time.time()
        time_elapsed = 0
        player.trust_param = trust

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

            delta_t = time.time() - prev_time
            time_elapsed += delta_t
            prev_time = time.time()

            player.update(delta_t)

            data.append([time_elapsed, player.x, player.y] + list(player.goal) + list(player.obs) + player.f1 + player.f2)
            pickle.dump(data, open(file_name, "wb" ))

            if player.dist2goal < 50:
                print(data)
                print("We made it to the target!")
                pygame.quit()
                break

            world.fill((0,0,0))
            sprite_list.draw(world)

            pygame.display.flip()
            clock.tick(fps)


if __name__ == "__main__":
    main()
