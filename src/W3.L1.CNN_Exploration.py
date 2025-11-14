import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt

def main():
    train, data = loadData()

    alex = AlexNET()
    print(alex)



    pass

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