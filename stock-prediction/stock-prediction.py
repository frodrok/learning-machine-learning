#!/bin/python3

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def download_data_and_save(stocks):


    # List of 8-10 stock tickers
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "JPM"]

    # Download data from Yahoo Finance
    print("Downloading")
    data = yf.download(stocks, start="2010-01-01", interval="1mo", auto_adjust=False)["Adj Close"]

    print("Finished downloading")
    # Compute monthly percentage change
    returns = data.pct_change()

    # Define the target variable (1 = Price went up, 0 = Price went down)
    labels = (returns > 0).astype(int)
    print(f"labels {labels}")

    # Combine returns and labels into a dataset
    dataset = returns.copy()

    for stock in stocks:
        dataset[f"{stock}_Target"] = labels[stock]

    # dataset["Target"] = labels

    # Save the dataset
    dataset.to_csv("stock_data.csv")
    print("Data saved to stock_data.csv")

def train_model(stocks, dataset):
    X = dataset[stocks]
    y = dataset[[f"{stock}_Target" for stock in stocks]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    for i, stock in enumerate(stocks):
        stock_acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"{stock} Accuracy: {stock_acc:.2f}")


stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "JPM"]
# download_data_and_save(stocks)
dataset = pd.read_csv("stock_data.csv", index_col=0)
dataset = dataset.dropna()


train_model(stocks, dataset)
