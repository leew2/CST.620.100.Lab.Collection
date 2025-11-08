import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    x, y = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    

    pass



def load_data():
    digits = datasets.load_digits(return_X_y=True)
    print_img(digits)
    x, y = digits
    
    return x, y

def print_img(img):
    plt.imshow(img[0], cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
    print("End of Code")