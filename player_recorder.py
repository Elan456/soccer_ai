"""
Allows the human player to control a player in the game and then all the states are recorded along with
x and y velocities of the player. The AI can then be trained to make the inputs which help it match the x and y
velocities of the human player at that same state.
"""
import pygame
import soccer
import math
import time
import pandas as pd
import numpy as np


def start():

    data = np.zeros((1, 12))

    pygame.init()
    pygame.display.set_caption("Soccer Game")
    screen = pygame.display.set_mode((soccer.FIELD_WIDTH, soccer.FIELD_HEIGHT))

    game = soccer.SoccerGame()
    game.players.append(soccer.Player(150, 150, 0))

    clock = pygame.time.Clock()
    for teleport in range(30):
        print("Teleport: " + str(teleport))
        # Moving the ball and the player to a random location on the field

        game.players[0].x = np.random.randint(0, soccer.FIELD_WIDTH)
        game.players[0].y = np.random.randint(0, soccer.FIELD_HEIGHT)

        game.ball.x = np.random.randint(0, soccer.FIELD_WIDTH)
        game.ball.y = np.random.randint(0, game.players[0].y)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            dxv, dyv = soccer.get_human_input()

            game.players[0].control_input(dxv, dyv)
            game.update(screen)
            pygame.display.flip()
            clock.tick(30)

            inputs_vals = soccer.collect_input_vals(game, 0)
            x_vel, y_vel = dxv, dyv

            # Rounding all the data to 4 decimal places
            inputs_vals = [round(x, 6) for x in inputs_vals]
            x_vel = round(x_vel, 6)
            y_vel = round(y_vel, 6)

            data = np.append(data, [[inputs_vals[0], inputs_vals[1], inputs_vals[2], inputs_vals[3], inputs_vals[4],
                                     inputs_vals[5], inputs_vals[6], inputs_vals[7], inputs_vals[8], inputs_vals[9], x_vel, y_vel]], axis=0)

            # If the player has scored, then break out of the loop
            if game.ball.team0_score > 0:
                game.ball.team0_score = 0
                break

    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False, header=False)

    pygame.quit()
    quit()


if __name__ == '__main__':
    start()




