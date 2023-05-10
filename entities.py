import pygame
import numpy

PLAYER_SIZE = 10
PLAYER_SPEED = 2

BALL_SIZE = 5
BALL_SPEED = 5

TEAM_COLORS = [(175, 0, 0), (0, 0, 175)]

GOAL_WIDTH = 50
FIELD_WIDTH = 300
FIELD_HEIGHT = 500

FRICTION = .1

def wall_collision(x, y):
    newx, newy = x, y
    if x < 0:
        newx = 0
    elif x > FIELD_WIDTH:
        newx = FIELD_WIDTH

    if y < 0:
        newy = 0
    elif y > FIELD_HEIGHT:
        newy = FIELD_HEIGHT

    return newx, newy



class Player:
    def __init__(self, x, y, team):
        self.x = x
        self.y = y
        self.team = team
        self.speed = PLAYER_SPEED

        self.xv = 0
        self.yv = 0

    def control_input(self, dxv, dyv):
        self.xv += dxv
        self.yv += dyv

        # If the cartesian velocity is greater than the speed, scale it down
        total_speed_sq = self.xv ** 2 + self.yv ** 2
        if total_speed_sq > self.speed ** 2:
            total_speed = total_speed_sq ** .5
            self.xv *= self.speed / total_speed
            self.yv *= self.speed / total_speed

    def move(self):
        self.x += self.xv
        self.y += self.yv

        # Apply friction
        self.xv *= 1 - FRICTION
        self.yv *= 1 - FRICTION

        # Check for wall collisions
        self.x, self.y = wall_collision(self.x, self.y)

    def draw(self, screen):
        pygame.draw.circle(screen, TEAM_COLORS[self.team], (int(self.x), int(self.y)), PLAYER_SIZE)


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.xv = 0
        self.yv = 0

        self.size = BALL_SIZE
        self.speed = BALL_SPEED

        self.team0_score = 0
        self.team1_score = 0

    def move(self, players, goals):
        good_kick = False
        # If the ball is touching a player, bounce off of them
        for p in players:
            if (self.x - p.x) ** 2 + (self.y - p.y) ** 2 < (self.size + PLAYER_SIZE) ** 2:
                # Find the angle between the ball and the player
                angle = numpy.arctan2(self.y - p.y, self.x - p.x)

                # Move the ball directly away from the player
                self.xv = numpy.cos(angle) * self.speed
                self.yv = numpy.sin(angle) * self.speed

                # Checking if the angle is towards the goal
                if self.yv > 0:
                    good_kick = True

                break

        # If the cartesian velocity is greater than the speed, scale it down
        total_speed_sq = self.xv ** 2 + self.yv ** 2
        if total_speed_sq > self.speed ** 2:
            total_speed = total_speed_sq ** .5
            self.xv *= self.speed / total_speed
            self.yv *= self.speed / total_speed

        # Applying friction
        self.xv *= 1 - FRICTION
        self.yv *= 1 - FRICTION


        self.x += self.xv
        self.y += self.yv

        # Wall collision check
        if self.x < 0:
            self.x = 0
            self.xv = -self.xv
        elif self.x > FIELD_WIDTH:
            self.x = FIELD_WIDTH
            self.xv = -self.xv

        if self.y < 0:
            self.y = 0
            self.yv = -self.yv
        elif self.y > FIELD_HEIGHT:
            self.y = FIELD_HEIGHT
            self.yv = -self.yv

        # Goal collision check
        for g in goals:
            if g.x < self.x < g.x + GOAL_WIDTH and g.y - self.size < self.y < g.y + self.size:
                self.x = FIELD_WIDTH / 2
                self.y = FIELD_HEIGHT / 2
                self.xv = 0
                self.yv = 0

                if g.team == 0:
                    self.team1_score += 1
                else:
                    self.team0_score += 1

        return good_kick

    def draw(self, screen):
        pygame.draw.circle(screen, (175, 0, 175), (int(self.x), int(self.y)), self.size)


class Goal:
    def __init__(self, team):
        self.team = team
        self.x = FIELD_WIDTH / 2 - GOAL_WIDTH / 2
        if team == 0:
            self.y = 0
        else:
            self.y = FIELD_HEIGHT - 1

    def draw(self, screen):
        pygame.draw.rect(screen, TEAM_COLORS[self.team], (self.x, self.y, GOAL_WIDTH, 1))

