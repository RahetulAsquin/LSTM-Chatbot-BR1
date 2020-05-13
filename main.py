from flask import Flask, render_template, request, jsonify
import aiml
import os
import re
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

df = pd.read_csv('monthly_data_till19.csv', parse_dates=[1], index_col=0, usecols=['ds','y'])

train = df

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)





app = Flask(__name__)


def pred(f1):
    flag=0
    n=0
    if f1.isdigit()==True:
        flag=1
    else:
        flag=0
    if flag==1:
        f=0
        f=int(f1)
        if f>0 :
            n_input = f
            n_features = 1
            generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

            model = Sequential()
            model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
            model.add(Dropout(0.15))
            model.add(Dense(1))
            optimizer = keras.optimizers.Adam(lr=0.001)
            model.compile(optimizer=optimizer, loss='mse')
            history = model.fit_generator(generator,epochs=100,verbose=1)

            pred_list = []
            batch = train[-n_input:].reshape((1, n_input, n_features))
            for i in range(n_input):
                pred_list.append(model.predict(batch)[0]) 
                batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

            import pandas as pd 
            ts = pd.Timestamp('2019-10-10 07:15:11') 
            do = pd.tseries.offsets.DateOffset(n = 2) 
            add_dates = [pd.Timestamp(df.index[-1]) + DateOffset(months=x) for x in range(0,f+1) ]
            future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)
            df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index, columns=['Prediction'])
            df_proj = pd.concat([df,df_predict], axis=1)
            res=df_predict.reset_index()
            res.columns=['Date','count of cases']
            res['count of cases'] = res['count of cases'].apply(np.int64)
            
            return res



	
@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
    userText = request.form['messageText']   
    ress= pred(userText)
    #ress=ress.to_json()
    print(userText)
    #predict_plot(userText)
    ress1 =ress.to_string(index=False)
    bot_response =ress1
    #predict_plot(userText)
    #print(bot_response)
    return jsonify({'status':'OK','answer':bot_response})


if __name__ == "__main__":
    app.run(host='127.0.0.4', debug=True)
