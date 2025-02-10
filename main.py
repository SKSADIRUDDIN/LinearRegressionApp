import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
def generate_data(n_samples=1000):
    np.random.seed(50)
    size = np.random.normal(1400,50,n_samples)
    price = size*50 + np.random.normal(0,50,n_samples)
    return pd.DataFrame({'size':size, 'price':price})

def train_model():
    df = generate_data(n_samples=1000)
    X=df[['size']]
    Y=df['price']
    X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    return model

def main():
    st.title("Simple Linear Rigression App for Predicting House Prices")

    st.write("Input the House Price ")
    model=train_model()
    size = st.number_input("House Size",min_value=100,max_value=2000,value = 100)
    if st.button("Predict Price"):
        pred_price=model.predict([[size]])
        st.success(f"Estimated Price is ${pred_price[0]:,.2f}")
        df=generate_data()
        fig = px.scatter(df,x="size",y="price",title="Size vs Price")
        fig.add_scatter(x=[size],y=[pred_price[0]],mode="markers",marker=dict(size=15,color="red"),name="Prediction")
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
