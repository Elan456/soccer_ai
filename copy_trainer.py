"""
Uses the first_data.csv with the input values and the x and y velocities of the player to train the model
Uses normal gradient descent to train the model, it doesn't actually play the game to train
"""

import torch
import soccermodel
import soccer
import random

# Load the data
data = []
with open("data.csv") as f:
    for line in f:
        data.append([float(v) for v in line.split(",")])

# Shuffle the data
random.shuffle(data)

# Load the model
try:
    model = soccermodel.SoccerModel()
    model.load_state_dict(torch.load("model.pt"))
    print("Loaded model")
    # Showing the model playing using the drill_fitness
except FileNotFoundError:
    model = soccermodel.SoccerModel()


soccer.play_against(model)
# soccer.verification(model, 100)

# Get an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

last_losses = []
for epoch in range(50):

    print("Epoch: " + str(epoch))
    # Taking a random sample of 10000 from the training data
    try:
        train_data = random.sample(data, 10000)
    except ValueError:
        train_data = data
    for i, d in enumerate(train_data):
        inputs = torch.tensor(d[:10], dtype=torch.float32)
        outputs = torch.tensor(d[10:], dtype=torch.float32) / 2

        # Forward pass
        pred = model(inputs)

        # print("Pred: " + str(pred))

        # Calculate the loss
        loss = (pred - outputs).pow(2).sum()


        # print("Loss: " + str(loss))
        # print("Inputs: " + str(inputs))
        # print("Outputs: " + str(outputs))
        # print("Pred: " + str(pred))

        last_losses.append(loss.item())
        if len(last_losses) > 1000:
            last_losses.pop(0)

        # Backpropagation
        loss.backward()

        # Update the weights
        optimizer.step()

        # Reset the gradients
        optimizer.zero_grad()

        if i % 1000 == 0:
            print("T_Loss: " + str(sum(last_losses) / len(last_losses)))


# Saving the model
torch.save(model.state_dict(), "model.pt")




