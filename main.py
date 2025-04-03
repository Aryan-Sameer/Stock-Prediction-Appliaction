from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/analyse', methods = ['POST'])
def analyze():
    ticker = request.form['ticker']
    period = request.form['period']

    data = yf.download(ticker, period=period)
    data['Moving Average'] = data['Close'].rolling(window=20).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Moving Average'], label='20-Day Moving Average')
    plt.title(f'{ticker} Stock Price Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('analysis.html', ticker=ticker, period=period, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
