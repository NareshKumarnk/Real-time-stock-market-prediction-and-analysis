#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask
from pandas_datareader import DataReader
from datetime import datetime as dt


# In[2]:


stock_lst = ['GOOG','TCS.NS','INFY','AMZN','MSFT','FB','IBM','DIS','AAPL','TSLA']
end_date = dt.now()
start_date = dt(end_date.year-20,end_date.month,end_date.day)
print(start_date ,'to', end_date)


# In[3]:


for stock in stock_lst:
    globals()[stock] = DataReader(stock,'yahoo',start_date,end_date)


# In[4]:


company_list=['GOOG','TCS.NS','INFY','AMZN','MSFT','FB','IBM','DIS','AAPL','TSLA']
for stock,company in zip(stock_lst,company_list):
    google_Stock = globals()[stock]
    stocks=pd.DataFrame(google_Stock)
    stocks['Stock']=company


# In[5]:


df = pd.DataFrame(columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Stock'])
for i in stock_lst:
    data = globals()[i]
    df = data.append(df)
df.reset_index(level=0, inplace=True)
df.rename(columns = {'index':'Date'}, inplace = True)


# In[6]:


df_nse = df[df['Stock']=='TCS.NS']


# In[7]:


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


# In[8]:


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values


# In[9]:


train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[10]:


model=load_model("model_bilstm.h5")
model1=load_model("saved_lstm_model.h5")
model2=load_model("model_gru.h5")


# In[11]:


def load_data(model):
    data=df_nse.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)
    dataset=new_data.values
    
    train=dataset[0:987,:]
    valid=dataset[987:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)
    
    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)
        
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)
    
    train=new_data[:987]
    valid=new_data[987:]
    valid['Predictions']=closing_price
    
    return valid


# In[12]:


train=new_data[:987]
valid=new_data[987:]
valid1 = load_data(model)
valid2 = load_data(model1)
valid3 = load_data(model2)


# In[13]:


app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='NS TCS Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid1.index,
								y=valid1["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
                html.H2("BiLSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data1",
					figure={
						"data":[
							go.Scatter(
								x=valid2.index,
								y=valid2["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
                html.H2("GRU Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data2",
					figure={
						"data":[
							go.Scatter(
								x=valid3.index,
								y=valid3["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)
			])        		


        ]),
        dcc.Tab(label='TCS Stock analysis', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Amazon', 'value': 'AMZN'},
                                      {'label': 'TCS', 'value': 'TCS.NS'},
                                      {'label': 'Google','value': 'GOOG'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Infosys', 'value': 'INFY'},
                                      {'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['TCS.NS'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Amazon', 'value': 'AMZN'},
                                      {'label': 'TCS', 'value': 'TCS.NS'},
                                      {'label': 'Google','value': 'GOOG'},
                                      {'label': 'Infosys', 'value': 'INFY'},
                                      {'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['TCS.NS'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TCS.NS":"TCS","AMZN": "Amazon","GOOG": "Google","FB": "Facebook","MSFT": "Microsoft","INFY":"Infosys","TSLA":"Tesla",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TCS.NS":"TCS","AMZN": "Amazon","GOOG": "Google","FB": "Facebook","MSFT": "Microsoft","INFY":"Infosys","TSLA":"Tesla",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
    #app.run_server(host='127.0.0.1', port=5000, debug=False)
    app.run_server(host='0.0.0.0',debug=False)
    


# In[ ]:




