import soccer
import pygame

screen = pygame.display.set_mode((soccer.FIELD_WIDTH, soccer.FIELD_HEIGHT))

ACTION_VEL_PAIRS = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
    4: (0, -1)
}

SHOW_STUFF = False

class SoccerGym:
    def __init__(self):
        self.game = None
        self.reset()

    def reset(self):
        self.game = soccer.SoccerGame()
        self.game.players.append(soccer.Player(150, 150, 0))

        return self.game.collect_input_vals(0)

    def step(self, action: int):
        """
        Converts the action to a dxv, dyv pair and calls the game's update method.
        :param action: The action to take (0-3)
        :return: observation, reward, done
        """
        action = ACTION_VEL_PAIRS[action]
        reward, done = self.game.update(action=action, screen=screen)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # To stop the game from going on forever
        if not done and self.game.ticks > 500:
            done = True

        observation = self.game.collect_input_vals(0)


        return observation, reward, done
