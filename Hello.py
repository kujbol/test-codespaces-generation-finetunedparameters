
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Displaying the versions of the dependencies
st.write('scikit-learn version:', sklearn.__version__)

# Load a dataset
def load_data(option):
    if option == 'Iris':
        data = datasets.load_iris()
    elif option == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    return data

# Split the dataset into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate a Logistic Regression model
def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Train and evaluate a k-Nearest Neighbors model
def knn(X_train, X_test, y_train, y_test, neighbors):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Train and evaluate a Decision Tree model
def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Sidebar
st.sidebar.title("scikit-learn Operations")

# Choose a dataset
dataset_option = st.sidebar.selectbox("Choose a dataset", ('Iris', 'Breast Cancer', 'Wine'))

data = load_data(dataset_option)
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Choose an operation
operation_option = st.sidebar.selectbox("Choose an operation", ('Logistic Regression', 'k-Nearest Neighbors', 'Decision Tree'))

# Perform the selected operation
if operation_option == 'Logistic Regression':
    accuracy = logistic_regression(X_train, X_test, y_train, y_test)
else:
    neighbors = st.sidebar.slider("Select the number of neighbors", min_value=1, max_value=10, value=5)
    if operation_option == 'k-Nearest Neighbors':
        accuracy = knn(X_train, X_test, y_train, y_test, neighbors)
    else:
        accuracy = decision_tree(X_train, X_test, y_train, y_test)

# Display the accuracy
st.write("Accuracy:", accuracy)
