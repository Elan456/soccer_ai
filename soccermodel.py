import torch.nn


class SoccerModel(torch.nn.Module):
    """
    Classification model for deciding what action to take in a soccer game
    Inputs:
        - Distance to the ball x
        - Distance to the ball y
        - Distance to the target goal x
        - Distance to the target goal y
        - Distance to the own goal x
        - Distance to the own goal y
        - Distance to bottom of the field
        - Distance to top of the field
        - Distance to left of the field
        - Distance to right of the field
    Outputs:
        - Move x (1 to -1) this is multiplied by the speed of the player to move left or right
        - Move y (1 to -1) this is multiplied by the speed of the player to move up or down
    """

    def __init__(self):
        super(SoccerModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 8)
        self.fc2 = torch.nn.Linear(8, 6)
        self.fc3 = torch.nn.Linear(6, 4)
        self.fc4 = torch.nn.Linear(4, 2)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        x = torch.nn.functional.tanh(x)
        return x

