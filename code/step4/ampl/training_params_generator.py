# Generates a json file that contains the parameters for a random parallel hpo with AMPL in IMPECCABLE

import random
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--node_count", required=True, type=int)
args = parser.parse_args()

# How many combinations need to be made
num_tasks = args.node_count * 8

# Layer size information
min_number_of_layers = 2
max_number_of_layers = 5
layer_sizes = [200, 400, 600, 800]

# Batch size range
batch_size_min = 600
batch_size_max = 1500

# Learning rate range
learning_rate_min = 0.0001
learning_rate_max = 0.0009

# Weight decay penalty options
weight_decay_penalty_types = ["l1", "l2"]

full_parameters = set()
for x in range(num_tasks):

    # Create a tuple that contains the layer sizes
    num_layers = random.randint(min_number_of_layers, max_number_of_layers)
    layers = tuple(random.choices(layer_sizes, k=num_layers))  # Use tuple for set storage
    
    # Define a tuple that contains the selected parameters
    param = (
        random.randint(batch_size_min, batch_size_max), # batch size
        layers, # layer size
        random.uniform(learning_rate_min, learning_rate_max), # learning rate
        random.choice(weight_decay_penalty_types)
    )

    # Add unique combinations of parameters to a set
    full_parameters.add(param)

# Convert the set to a list to enumerate over
parameters_list = list(full_parameters)

# Creates a dictionary where the key is the index and the value is the tuple of parameters
parameters_dict = {
    index: param for index, param in enumerate(parameters_list)
}

# Dumps the dictionary into a json file in current working directory
with open("training_params.json", "w") as f:
    json.dump(parameters_dict, f, indent=4)