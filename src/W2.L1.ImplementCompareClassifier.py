import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def main():
    x, y = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    classifier = implement_classifier(x_train, y_train)
    metrics("KNN", classifier, x_test, y_test)
    svm_classifier = implement_svm_classifier(x_train, y_train)
    metrics("SVM", svm_classifier, x_test, y_test)
    log_reg_classifier = implement_logistic_regression_classifier(x_train, y_train)
    metrics("Logistic Regression", log_reg_classifier, x_test, y_test)
    best_knn = GridSearchCV_model(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}, x_train, y_train)
    metrics("Best KNN after Grid Search", best_knn, x_test, y_test)


# Grid Search CV -----------------------------------------------------------------------------
def GridSearchCV_model(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

# Metrics ---------------------------------------------------------------------------------
def metrics(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
# End Metrics ------------------------------------------------------------------------------

# Implement Classifier ---------------------------------------------------------------------
def implement_classifier(x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x_train, y_train)
    return classifier

def implement_svm_classifier(x_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(x_train, y_train)
    return svm_classifier

def implement_logistic_regression_classifier(x_train, y_train):
    log_reg_classifier = LogisticRegression(max_iter=1000)
    log_reg_classifier.fit(x_train, y_train)
    return log_reg_classifier

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