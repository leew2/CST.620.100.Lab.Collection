import torch
import torch.nn as nn   
import torch.optim as optim
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    print(f"Dataset Load \nX Tensor:{x_train_tensor.shape}\nY Tensor:{y_train_tensor.shape}")
    
    pass

# load ----------------------------------------------------------------------------------
def get_data():
    x,y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    x = x.reshape(x.shape[0], -1)
    x = scaler.fit_transform(x)
    return x, y
# end of load ---------------------------------------------------------------------------

if __name__ == main():
    main()
    print("Task Done")