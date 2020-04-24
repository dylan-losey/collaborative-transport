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
        return [fx*50, fy*50], e_stop


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
        goal_angle = math.pi/4#np.random.uniform(-math.pi, math.pi)
        goal_radius = 100
        self.goal = np.array([0.5*worldx, 0.5*worldy])
        self.goal += goal_radius * np.array([math.cos(goal_angle), math.sin(goal_angle)])
        self.dist2goal = None

        # create obstacle
        obs_angle = goal_angle + np.random.uniform(-math.pi/6, math.pi/6)
        obs_radius = 0
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
        self.x = 0.25*worldx
        self.y = 0.25*worldy
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
        self.sigma = 0.0

    def acceleration(self, f1, f2):

        # equations of motion for table
        a_x = -b/m * self.x_speed + 1/m * (f1[0] + f2[0])
        a_y = -b/m * self.y_speed + 1/m * (f1[1] + f2[1])
        M_z = L / 2.0 * (math.sin(self.angle)*(f1[0] - f2[0]) - math.cos(self.angle)*(-f1[1] + f2[1]))
        a_angle = -d/I * self.angle_speed + 1/I * M_z
        return a_x, a_y, a_angle

    def robot(self):

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
            magnitude = -0.001*(150.0 - self.dist2obs)**2
        fobs = magnitude * avoid

        # choose robot force
        f1 = fgoal + fobs
        force_magnitude = np.linalg.norm(f1)
        if force_magnitude > 50:
            f1 *= 50 / force_magnitude
        f1 += np.random.normal(self.mu, self.sigma, 2)
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

    num_of_runs = int(sys.argv[1])
    fps = 40

    for count in range(num_of_runs):

        file_name = "experimental-data/r" + str(count) + ".pkl"
        print("round: ", count)

        pygame.init()
        clock = pygame.time.Clock()
        world = pygame.display.set_mode([worldx,worldy])

        player = Player()
        joystick = Joystick()
        target = Target(player.goal)
        obs = Obstacle(player.obs)
        sprite_list = pygame.sprite.Group()
        sprite_list.add(target)
        sprite_list.add(obs)
        sprite_list.add(player)

        data = []
        prev_time = time.time()

        while True:

            delta_t = time.time() - prev_time
            prev_time = time.time()

            f2, e_stop = joystick.input()
            if e_stop:
                pygame.quit(); sys.exit()
                main = False

            player.robot()
            player.f2 = f2
            player.update(delta_t)

            data.append([player.x, player.y] + list(player.goal) + player.f1 + player.f2 + list(player.obs))
            # pickle.dump(data, open(file_name, "wb" ))

            if player.dist2goal < 50:
                pygame.quit()
                break

            world.fill((0,0,0))
            sprite_list.draw(world)
            net_display(world, player.f1, player.x, player.y)

            pygame.display.flip()
            clock.tick(fps)


if __name__ == "__main__":
    main()
