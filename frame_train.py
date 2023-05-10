"""
TicTacToe Ai
First random moves will be made for both players for 1000 games every action, state, reward will be stored in a list
Then an AI will be trained to calculate the reward given a state.
This AI can then be used to play against a human player
"""

import numpy as np
from tictactoe_gym import TicTacToeGym
import torch
import torch.nn
import torch.optim
import time

gamma = 0.75

class RewardPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(9, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def choose_move(self, board):
        """
        Tries every possible move and chooses the one which creates a board with the highest reward
        """

        rewards = []
        for i in range(9):
            if board[i] == 0:
                new_board = board.clone()
                new_board[i] = 1
                reward = self(new_board).detach().numpy()[0]
                rewards.append(reward)
            else:
                rewards.append(-100)
        return np.argmax(rewards)


def get_train_data():
    """
    Plays the game 1000 times with both players making random moves and then saves all the data as a csv
    When a game ends, all the moves are given the same reward dropped off: 1 if X wins, -1 if O wins, 0 if it is a tie
    """

    all_states = []
    all_actions = []
    all_rewards = []

    env = TicTacToeGym()
    x_wins = 0
    for game in range(10000):
        states = []
        actions = []
        rewards = []

        # Resets the board
        env.reset()
        done = False
        while not done:
            action = env.get_random_move()
            observation, reward, done = env.step(None)
            # env.render()

            states.append(observation)  # This is the state after the action has been made
            actions.append(action)
            rewards.append(reward)  # Will be 0 unless the game is over

        final_reward = rewards[-1]
        if final_reward == 1:
            x_wins += 1

        # Going through the rewards to calculate the discounted rewards
        for i in range(len(rewards)):
            distance = len(rewards) - i - 1
            rewards[i] = final_reward * gamma ** distance  # The further away the reward is, the less this action mattered


        all_states += states
        all_actions += actions
        all_rewards += rewards

    print(all_rewards)
    print("x wins: ", x_wins)

    # Saving the data together with the states, actions and rewards
    np.savez('train_data', states=all_states, actions=all_actions, rewards=all_rewards)


def train_model_from_data():
    model = RewardPredictor()
    data = np.load('train_data.npz')

    # Taking 80% of the data for training and 20% for testing
    train_size = int(len(data['states']) * 0.8)

    train_states = torch.tensor(data['states'][:train_size], dtype=torch.float32)
    train_rewards = torch.tensor(data['rewards'][:train_size], dtype=torch.float32)

    test_states = torch.tensor(data['states'][train_size:], dtype=torch.float32)
    test_rewards = torch.tensor(data['rewards'][train_size:], dtype=torch.float32)

    for epoch in range(1000):
        # Training the model
        output = model(train_states).squeeze()
        loss = torch.nn.functional.mse_loss(output, train_rewards)
        print("Epoch: ", epoch, "Loss: ", loss.item())

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    # Testing the model
    output = model(test_states).squeeze()
    loss = torch.nn.functional.mse_loss(output, test_rewards)
    print("Test loss: ", loss.item())


    torch.save(model.state_dict(), 'reward_predictor.pt')


def train_from_play():
    """
    The Ai will play against the semi-random player then the loss will be calculated using the game's conclusion
    """

    model = RewardPredictor()
    epsilon = 0.9

    # Many games will be played
    for _ in range(10000):
        env = TicTacToeGym()
        env.reset()
        states = []
        rewards = []

        done = False
        while not done:
            # The Ai will choose the best move or a random move
            if np.random.random() < epsilon:
                action = env.get_random_move()
            else:
                observation = torch.tensor(env.board, dtype=torch.float32)
                action = model.choose_move(observation)

            observation, reward, done = env.step(action)

            states.append(observation)
            rewards.append(reward)

        # Going through the rewards to calculate the discounted rewards
        for i in range(len(rewards)):
            distance = len(rewards) - i - 1
            rewards[i] = reward * gamma ** distance

        # Training the model
        output = model(torch.tensor(states, dtype=torch.float32)).squeeze()
        loss = torch.nn.functional.mse_loss(output, torch.tensor(rewards, dtype=torch.float32))
        print("Loss: ", loss.item())

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    torch.save(model.state_dict(), 'reward_predictor.pt')



def verify_model():
    """
    Play against random moves for 100 games and see how many games the model wins
    :return:
    """
    # loads the model from the file
    model = RewardPredictor()
    model.load_state_dict(torch.load('reward_predictor.pt'))

    # creates a new game
    env = TicTacToeGym()
    env.reset()
    done = False
    wins = 0
    losses = 0
    ties = 0
    for game in range(1000):
        env.reset()
        done = False
        while not done:
            # gets the action that the model thinks is best

            observation = torch.tensor(env.board, dtype=torch.float32)
            action = model.choose_move(observation)
            observation, reward, done = env.step(action)
            env.render()

            time.sleep(.5)
        time.sleep(.3)

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            ties += 1
    print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')


def human_play():
    # Loads the model
    model = RewardPredictor()
    model.load_state_dict(torch.load('reward_predictor.pt'))

    # Creates a new game
    while True:
        env = TicTacToeGym()
        env.reset()
        done = False
        t = -1
        while not done:
            env.render()
            t += 1
            # Gets the action that the model thinks is best
            if t % 2 == 0:
                observation = torch.tensor(env.board, dtype=torch.float32)
                action = model.choose_move(observation)
                observation, reward, done = env.step(action, do_O=False)
                time.sleep(.3)
            else:
                env.get_user_move()

            if env.check_win() != 0:
                done = True

        env.render()
        time.sleep(.3)

train_from_play()
human_play()
# get_train_data()
# verify_model()
# train_model()