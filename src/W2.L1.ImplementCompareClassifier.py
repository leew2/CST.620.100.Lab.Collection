import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def main():
    x, y = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    classifier = implement_classifier(x_train, y_train)
    metrics("KNN", classifier, x_test, y_test)

def metrics(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Implement Classifier ---------------------------------------------------------------------
def implement_classifier(x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x_train, y_train)
    return classifier


# End Implement Classifier -----------------------------------------------------------------

# Load and view ------------------------------------------------------------------------------
def load_data():
    digits = datasets.load_digits(return_X_y=True)
    #print_img(digits)
    x, y = digits
    
    return x, y

def print_img(img):
    plt.imshow(img[0], cmap="gray")
    plt.axis("off")
    plt.show()
# End Load and view --------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("End of Code")