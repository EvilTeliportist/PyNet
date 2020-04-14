import os
import matplotlib.pyplot as plt
from Robinhood import Robinhood
import numpy
import yfinance as yf

def get_data(name):
    
    ticker = yf.Ticker(name)
    hist = ticker.history(period='max')
    hist2 = [i[1] for i in hist.to_numpy()]
    return hist2
    

def make_ema(data, smoothing):
    ema = [data[0]]

    for i in range(len(data))[1:]:
        ema.append((smoothing*data[i]) + ((1 - smoothing) * ema[-1]))

    return ema

def find_periods(ema, dema, neg_thresh):
    above = True
    points = []
    for i in range(len(ema)):
        if (ema[i] < dema[i]) == above:
            points.append(i)
            above = not above

    points = points[::2]

    slope = []
    last = 0
    for i in range(len(dema)):
        if dema[i] < (last - neg_thresh):
            slope.append(i-1)
        last = dema[i]

    return points, slope

def cutoff(index, neg):
    for i in neg:
        if i > index:
            return i

def make_graph(name, day_thresh, neg_thresh):
    clean = get_data(name)#'C:/Users/brhou/Desktop/trading_bot/BackTesting/historical_data/' + name)
    data = clean#[float(i[4]) for i in clean[1:]]

    #higher smoothness less smooth
    ema = make_ema(data, .1)
    dema = make_ema(data, .25)
    buy, slope = find_periods(ema, dema, neg_thresh)
    spans = [[i, cutoff(i, slope)] for i in buy]

    plt.plot(data)
    plt.plot(ema)
    plt.plot(dema)


    profits = 0
    for i in spans:
        try:
            if i[1] - i[0] > day_thresh:
                plt.axvspan(i[0], i[1]+1, color = 'red', alpha = .5)
                profits += (data[i[1]] - data[i[0]])
        except:
            pass

    print('Total Proft: $' + str(round(profits, 2)) + ' or ' + str(round(profits/data[0], 2))
          + '%\nAnnualized Return: $' + str(round(profits/(len(data)/253), 2)))

    plt.show()

make_graph('TSLA', 0, -1 )
