import pygad.torchga
from pygad.torchga import TorchGA
from pygad.torchga import torchga
import torch
import numpy
from soccermodel import SoccerModel
from soccer import drill_fitness

# ga_instance = pygad.GA(num_generations=1000,
#                        num_parents_mating=20,
#                        fitness_func=fitness_func,
#                        sol_per_pop=50,
#                        num_genes=4,)
#
# ga_instance.run()
# ga_instance.plot_result()


model = SoccerModel()

test_data = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

fitness_func_calls = 0
highest_fitness = float("-inf")

def fitness_func(solution, solution_idx):
    global model
    global fitness_func_calls
    global highest_fitness

    fitness_func_calls += 1

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    fitness = drill_fitness(model, show=False)  # Has the model play with the ball and returns the fitness

    if fitness > highest_fitness and fitness_func_calls > 10:
        fitness_func_calls = 0
        highest_fitness = fitness
        print("Highest fitness: " + str(highest_fitness))

        # Save the model weights to a file
        torch.save(model.state_dict(), "model.pt")


    return fitness


torch_ga = TorchGA(model=model,
                  num_solutions=50)

# Loading in the best model and putting it into the starting population
try:
    model.load_state_dict(torch.load("model.pt"))
    for i in range(50):
        torch_ga.population_weights[i] = torchga.model_weights_as_vector(model=model)
except FileNotFoundError:
    print("No best model found, starting with random weights")



# The ga actually runs the algorithm on the torch_ga
ga_instance = pygad.GA(num_generations=10000,
                       num_parents_mating=3,
                       initial_population=torch_ga.population_weights,
                       fitness_func=fitness_func,
                       mutation_probability=.5,
                       keep_parents=2,

                       )

ga_instance.run()
ga_instance.plot_result()
