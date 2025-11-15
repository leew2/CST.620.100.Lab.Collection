import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights


def main():
    train, data = loadData()

    alex = AlexNET()
    res = load_resNet()

    crit = nn.CrossEntropyLoss()
    trainer(model=alex, loader=train, criterion=crit)
    trainer(model=res, loader=train, criterion=crit)
    
    pass

# ResNet ------------------------------------------------------------------------
def load_resNet():
    res = resnet18(weights=ResNet18_Weights.DEFAULT)
    return res
# -------------------------------------------------------------------------------

# CNN ---------------------------------------------------------------------------
class AlexNET(nn.Module):
    def __init__(self, num_class=10,):
        super(AlexNET, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
# -------------------------------------------------------------------------------

# Train -------------------------------------------------------------------------
def trainer(model, loader, criterion, epochs=2): # 2 due to hardware limitation
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    counter = 0
    for epoch in range(epochs):
        for img, labels in loader:
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            if counter % 100 ==0: # print every 100 data point
                print(f'Trained {counter} data, Loss: {loss.item():.4f}')
            counter +=1

# -------------------------------------------------------------------------------

# Load Data ---------------------------------------------------------------------
def loadData():
    myTransform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.5), (0.5))
    ])
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=myTransform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=myTransform)
    showImg(train)
    train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    
    return train, test

def showImg(imgs):
    plt.imshow(imgs.data[0])
    plt.title(f"Sample Image - Class/Label: {imgs.targets[0]}")
    plt.show()
        



# -------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

    print('CNN Exploration Done')