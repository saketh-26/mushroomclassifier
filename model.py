import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.metrics import precision_score,recall_score

st.set_option('deprecation.showPyplotGlobalUse',False)
def main():
    st.title("Mushroom Classifier")
    st.sidebar.title("App Sidebar")
    st.sidebar.markdown("Let's start")

    @st.cache(persist = True)
    def load(): #DataLoading
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for i in data.columns:
            data[i] = label.fit_transform(data[i])
        return data
    df = load()
    if st.sidebar.checkbox("Display data",False):
        st.subheader("Show dataset")
        st.write(df)

    @st.cache(persist =True)
    def split(df):
        y = df['class']
        x = df.drop(columns=["class"])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test = split(df)

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model,x_test,y_test)
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model,x_test,y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model,x_test,y_test)
            st.pyplot()
    class_names = ['edible','poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Logistic Regression","Random Forest"))

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,
                                    10.0,step=0.01,key = "C_LR")
        max_iter = st.sidebar.slider("Maximum iterations",100,500,key = "max_iter")
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ("Confusion Matrix","ROC Curve",
                                          "Precision-Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:",accuracy.round(2))
            st.write("Precision:",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall:",recall_score(y_test,y_pred,
                                            labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the Forest",100,5000,step=10,key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree",
                                            1,20,step=1,key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when bulilding trees",
                                     (True,False),key="bootstrap")
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ("Confusion Matrix","ROC Curve",
                                          "Precision-Recall Curve"))
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth = max_depth,
                                           bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:",accuracy.round(2))
            st.write("Precision:",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall:",recall_score(y_test,y_pred,
                                            labels = class_names).round(2))
            plot_metrics(metrics)
        
    
if __name__ == "__main__":
    main()
    
