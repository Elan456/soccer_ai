import numpy as np
from gym import Gym
import pygame

screen = pygame.display.set_mode((300, 300))


class TicTacToeGym(Gym):
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        return self.board

    def step(self, action, do_O=True):
        """
        The action is the move made by X, then O makes a random move
        :param do_O:
        :param action: A single integer from 0-8

        :return: observation, reward, done
        """

        if action is None:
            self.semi_random_move(1)
        else:
            # Checks if the move is valid
            if self.board[action] != 0:
                # If the move is invalid, the game is over and X loses
                return self.board, -1, True

            # Makes the move
            self.board[action] = 1

        # Checks if the game is over
        win = self.check_win()

        if win == 0 and do_O:
            # If the game is not over, O makes a random move, although it will make a winning move if it can and will block X if it can
            self.semi_random_move(-1)
            win = self.check_win()

        # Determining the reward
        if win == 1:
            reward = 1
        elif win == -1:
            reward = -1
        else:
            reward = 0

        # Determining if the game is over
        done = win != 0

        return self.board, reward, done

    def check_win(self):
        """
        Checks if the game is over
        :return: 1 if X won, -1 if O won, 0 if no one won, 2 if it is a tie
        """
        # Checks for a horizontal win
        for i in range(3):
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                return self.board[i * 3]

            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                return self.board[i]

        # Checks for a diagonal win
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != 0:
            return self.board[2]

        # Checks if the board is full
        if np.all(self.board != 0):
            return 2

        return 0

    def get_random_move(self):
        return np.random.choice(np.where(self.board == 0)[0])

    def random_move(self, player=-1):
        """
        Makes a random move for O
        :return: None
        """
        # Gets a list of all the valid moves
        valid_moves = np.where(self.board == 0)[0]
        # Chooses a random move
        move = np.random.choice(valid_moves)
        # Makes the move
        self.board[move] = player

    def semi_random_move(self, player=-1):
        """
        If it can make a winning move, it will, otherwise it will make a random move
        :param player: The player to make the move for
        :return: None
        """

        # Gets a list of all the valid moves
        valid_moves = np.where(self.board == 0)[0]

        # Checks if it can win
        for move in valid_moves:
            self.board[move] = player
            if self.check_win() == player:
                return
            self.board[move] = 0

        # Checks if the opponent can win
        for move in valid_moves:
            self.board[move] = -player
            if self.check_win() == -player:
                self.board[move] = player
                return
            self.board[move] = 0

        # Chooses a random move
        move = np.random.choice(valid_moves)
        # Makes the move
        self.board[move] = player

    def same_move(self):
        """
        Makes a random move for O
        :return: None
        """
        # Gets a list of all the valid moves
        valid_moves = np.where(self.board == 0)[0]
        # Chooses a random move
        move = valid_moves[0]
        # Makes the move
        self.board[move] = -1

    def render(self):
        """
        Renders the game
        :param screen: The pygame screen
        :return: None
        """
        screen.fill((255, 255, 255))
        # Draws the lines
        pygame.draw.line(screen, (0, 0, 0), (100, 0), (100, 300), 5)
        pygame.draw.line(screen, (0, 0, 0), (200, 0), (200, 300), 5)
        pygame.draw.line(screen, (0, 0, 0), (0, 100), (300, 100), 5)
        pygame.draw.line(screen, (0, 0, 0), (0, 200), (300, 200), 5)
        # Draws the X's and O's
        for i in range(3):
            for j in range(3):
                if self.board[i * 3 + j] == 1:
                    pygame.draw.line(screen, (0, 0, 0), (j * 100 + 10, i * 100 + 10), (j * 100 + 90, i * 100 + 90), 5)
                    pygame.draw.line(screen, (0, 0, 0), (j * 100 + 90, i * 100 + 10), (j * 100 + 10, i * 100 + 90), 5)
                elif self.board[i * 3 + j] == -1:
                    pygame.draw.circle(screen, (0, 0, 0), (j * 100 + 50, i * 100 + 50), 40, 5)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


    def get_user_move(self):
        """
        Gets the user's move
        :return: None
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    x = pos[0] // 100
                    y = pos[1] // 100
                    if self.board[y * 3 + x] == 0:
                        self.board[y * 3 + x] = -1
                        return
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()


    def nice_print(self):
        """
        Prints the board in a nice format
        :return: None
        """
        print(self.board.reshape((3, 3)))
