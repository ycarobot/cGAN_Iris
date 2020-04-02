from sklearn.datasets import load_iris
import json
import numpy as np
import torch

data = load_iris()

X = data.data.tolist()
y = data.target.tolist()


filename = "iris_Dataset_data.json"
with open(filename, "w") as file_obj:
    json.dump(X, file_obj)

filename = "iris_Dataset_label.json"
with open(filename, "w") as file_obj:
    json.dump(y, file_obj)

# with open("iris_Dataset_data.json") as file_obj:
#   numbers = json.load(file_obj)
# print(len(numbers))
#
# with open("iris_Dataset_label.json") as file_obj:
#   numbers = json.load(file_obj)
# print(len(numbers))