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

    model = Model(input=x_train.shape[1], hidden=10, output=len(set(y)))
    trainNN(model, x_train=x_train_tensor, y_train=y_train_tensor, crit=nn.CrossEntropyLoss(), opti=optim.Adam(model.parameters(), lr=0.01), ep=100)

    with torch.no_grad():
        y_pred = model(x_test_tensor)
        y_pred_lab = torch.argmax(y_pred, dim=1)
    acc = (y_pred_lab == y_test_tensor).sum().item()/len(y_test_tensor)
    print(f'Model Accuracy: {acc:.4f}')
    
    plt.figure(figsize=(8, 4))
    plt.bar(["Setosa", "Versicolor", "Virginica"], [int((y_pred_lab == i).sum()) for i in range(3)])
    plt.xlabel("Class")
    plt.ylabel("Predicted Count")
    plt.title("Predicted Class Distribution")
    plt.show()


    pass

# train ------------------------------------------------------------------------
def trainNN(model, x_train, y_train, crit, opti, ep=100):
    for amount in range(ep):
        opti.zero_grad()
        outputs = model(x_train)
        loss = crit(outputs, y_train)
        loss.backward()
        opti.step()

        if amount % 10 == 0:
            print(f"Epoch {amount}: Loss={loss.item():.4f}")

# end train --------------------------------------------------------------------

# model ---------------------------------------------------------------------
def Model(input, hidden, output):
    model = twoLayerNN(input_size=input, hidden_size=hidden, output_size=output)
    print("Model:", model)
    return model

class twoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(twoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x
# end model -----------------------------------------------------------------

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