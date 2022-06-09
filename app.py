import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit Tutorial")

st.write("""
# Explore different classifier
Which one is the best? 
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ["iris", "wine", "breast_cancer"])

classifier_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "Decision Tree", "Random Forest"])

def get_dataset(data_name):
    if data_name == "iris":
        data = datasets.load_iris()
    elif data_name == "wine":
        data = datasets.load_wine()
    elif data_name == "breast_cancer":
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

st.write("Shape of the dataset: ", X.shape)
st.write("Shape of the target: ", y.shape)

def add_perimeter_ui(clf_name):
  params = dict()
  if clf_name == "Logistic Regression":
    C = st.sidebar.slider("C", 0.01, 10.0)
    params["C"] = C
  elif clf_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20)
    params["max_depth"] = max_depth
  elif clf_name == "Random Forest":
    max_depth = st.sidebar.slider("Max Depth", 1, 20)
    n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    params["max_depth"] = max_depth
    params["n_estimators"] = n_estimators
  return params

params = add_perimeter_ui(classifier_name)

def get_classifier(clf_name, params):
  if clf_name == "Logistic Regression":
    clf = LogisticRegression(**params)
  elif clf_name == "Decision Tree":
    clf = DecisionTreeClassifier(**params)
  elif clf_name == "Random Forest":
    clf = RandomForestClassifier(**params)
  return clf

# Classification

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write("Classifier: ", classifier_name)
st.write("Accuracy: ", acc) 

# PLOT
pca = PCA(2)
x_projected = pca.fit_transform(X)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(plt)