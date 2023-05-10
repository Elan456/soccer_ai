import torch
import numpy
import math
from pygad.torchga import torchga
from soccermodel import SoccerModel
import pygame
from entities import *
import numpy as np

MAX_DISTANCE = math.sqrt(FIELD_WIDTH ** 2 + FIELD_HEIGHT ** 2)

SHOW_GAME = True

screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))
pygame.display.set_caption("Soccer Game")

from soccermodel import SoccerModel


def drill_fitness(agent: SoccerModel, show=SHOW_GAME, frames=500):
    """Measures fitness by how many goals the agent can score within a set number of update cycles
        Also they are given a small amount of fitness based on how close the ball is to the goal at the end
    """

    game = SoccerGame()
    game.players.append(Player(150, 150, 0))
    game.add_AI_player(agent, 0)

    clock = pygame.time.Clock()
    for _ in range(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        if show:
            game.update(screen)
            pygame.display.flip()
            clock.tick(60)
        else:
            game.update()

    fitness = 0
    fitness += game.ball.team0_score
    ball_distance_to_goal = ((game.ball.x - game.goals[1].x) ** 2 + (game.ball.y - game.goals[1].y) ** 2) ** .5
    player_distance_to_ball = math.dist((game.ball.x, game.ball.y), (game.players[0].x, game.players[0].y))

    fitness -= (ball_distance_to_goal / MAX_DISTANCE) * 2

    # Having the player close to the ball
    # fitness -= (player_distance_to_ball / MAX_DISTANCE)

    # print("Fitness: " + str(fitness), "Score: " + str(game.ball.team0_score))
    return fitness


def verification(model, length=500):
    """
    Has the player start in a few locations to see how good it is at scoring
    """

    game = SoccerGame()
    game.players.append(Player(150, 150, 0))
    game.add_AI_player(model, 0)

    clock = pygame.time.Clock()
    for teleport in range(3):
        # Moves the player to a random location
        game.players[0].x = np.random.randint(0, FIELD_WIDTH)
        game.players[0].y = np.random.randint(0, FIELD_HEIGHT)
        for _ in range(length):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            game.update(screen)
            pygame.display.flip()
            clock.tick(60)


def play_against(model):
    """
    Has a human play against the AI for fun
    """

    game = SoccerGame()
    game.players.append(Player(150, FIELD_WIDTH / 2 - 100, 0))
    game.add_AI_player(model, 0)

    game.players.append(Player(150,FIELD_WIDTH / 2 + 100, 1))
    game.set_human_control(1)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        game.update(screen)
        pygame.display.flip()
        clock.tick(60)





class AiPlayer:
    def __init__(self, model, player_num):
        self.model = model
        self.player_num = player_num


def get_human_input():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        dxv = -1
    elif keys[pygame.K_RIGHT]:
        dxv = 1
    else:
        dxv = 0

    if keys[pygame.K_UP]:
        dyv = -1
    elif keys[pygame.K_DOWN]:
        dyv = 1
    else:
        dyv = 0

    dxv *= PLAYER_SPEED
    dyv *= PLAYER_SPEED

    return dxv, dyv


class SoccerGame:
    def __init__(self):
        self.players = []
        self.ball = Ball(FIELD_WIDTH / 2, FIELD_HEIGHT / 2)
        self.goals = [Goal(0), Goal(1)]
        self.ai_controllers = []
        self.human_control = None

        self.ticks = 0

    def add_AI_player(self, model: SoccerModel, player_num):
        self.ai_controllers.append(AiPlayer(model, player_num))

    def set_human_control(self, player_num):
        self.human_control = player_num

    def collect_input_vals(self, player_num):
        p = self.players[player_num]

        target_goal = self.goals[1 - p.team]
        own_goal = self.goals[p.team]

        input_vals = [
            # Distance to ball
            p.x - self.ball.x,
            p.y - self.ball.y,
            # 0,0,0,0,0,0,0,0
            # Distance to target goal
            p.x - target_goal.x,
            p.y - target_goal.y,
            # Distance to own goal
            p.x - own_goal.x,
            p.y - own_goal.y,
            # Distance to the bottom wall
            p.y - FIELD_HEIGHT,
            # Distance to the top wall
            p.y,
            # Distance to the left wall
            p.x,
            # Distance to the right wall
            p.x - FIELD_WIDTH
        ]
        input_vals = [v / FIELD_HEIGHT for v in input_vals]

        return input_vals


    def update(self, screen=None, action=None):
        self.ticks += 1
        # Draw the field
        if screen is not None:
            pygame.draw.rect(screen, (0, 175, 0), (0, 0, FIELD_WIDTH, FIELD_HEIGHT))

            # Draw the goals
            for g in self.goals:
                g.draw(screen)
        if action is None:  # If no action is given, collect input data and use the model to get an action
            for c in self.ai_controllers:
                p = self.players[c.player_num]
                # Collect input values
                input_vals = self.collect_input_vals(c.player_num)

                # Convert to a tensor
                input_vals = torch.tensor(input_vals, dtype=torch.float32)

                # Run the model
                output_vals = c.model(input_vals)

                # Convert to numpy array
                output_vals = output_vals.detach().numpy()
                # Multiply by the speed of the player
                output_vals *= PLAYER_SPEED
                # print("Output: " + str(output_vals))

                # Move the player
                p.control_input(output_vals[0], output_vals[1])

            if self.human_control is not None:
                dxv, dyv = get_human_input()
                self.players[self.human_control].control_input(dxv, dyv)
        else:
            # An action was given probably for q-learning
            # This action will be used to control the one and only player
            self.players[0].control_input(action[0], action[1])

        for p in self.players:
            p.move()
            if screen is not None:
                p.draw(screen)

        good_kick = self.ball.move(self.players, self.goals)

        if screen is not None:
            self.ball.draw(screen)

        if action is not None:
            # A reward has to be returned and whether the game is over
            # When a goal is scored, a reward of 1 is returned and the game is over
            # When a goal is scored against, a reward of -1 is returned and the game is over
            # When the game is not over, a reward of 0 is returned
            if self.ball.team0_score > 0:
                return 1, True
            elif self.ball.team1_score > 0:
                return -1, True
            else:
                # If the ball just got kicked closer to the goal, then a reward of 0.1 is returned
                # Otherwise, a reward of 0 is returned
                if good_kick:
                    return 0.1, False
                else:
                    return 0, False


if __name__ == "__main__":
    game = SoccerGame()
    game.players.append(Player(100, 100, 0))
    game.ball = Ball(150, 150)

    clock = pygame.time.Clock()

    dxv = 0
    dyv = 0

    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            dxv = -.1
        elif keys[pygame.K_RIGHT]:
            dxv = .1
        else:
            dxv = 0

        if keys[pygame.K_UP]:
            dyv = -.1
        elif keys[pygame.K_DOWN]:
            dyv = .1
        else:
            dyv = 0

        game.players[0].control_input(dxv, dyv)

        screen.fill((255, 255, 255))
        game.update(screen)
        pygame.display.update()
        clock.tick(60)
