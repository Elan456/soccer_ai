from soccer_gym import SoccerGym
from tictactoe_gym import TicTacToeGym
from q_soccer_model import Agent
import numpy as np
import torch
import time

if __name__ == '__main__':
    env = TicTacToeGym()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=9, eps_end=0.05,
                  input_dims=[9], lr=0.00001)
    scores, eps_history = [], []
    n_games = 10000

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            # print("action: ", action)
            observation_, reward, done = env.step(action)
            env.render()
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        if i % 10 == 0:
            # Saving the model every 10 episodes, without using the member function
            torch.save(agent.Q_eval.state_dict(), 'q_soccer_model.pt')

