import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt

def main():
    loadData()




    pass

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