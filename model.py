import joblib
from flask import Flask, jsonify, request, redirect, url_for, render_template
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings 
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
import flask
from datetime import timedelta
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'output_image')
app.config["DEBUG"] = True
path = os.getcwd()+'\\trained_models'

featureList = ['exec_ID','start_time','run_time', 'pass']

# Retrieve training model information
modelrf = joblib.load(path+'\\rfmul.pkl')    
scaler = joblib.load(path+'\\scaler.pkl')
@app.route('/', methods=['GET'])
def index():
    return flask.render_template('home.html')

def uploadFiles():    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        global file_path
        file_path = os.getcwd()+'\\input_files\\'+uploaded_file.filename
        uploaded_file.save(file_path)
    return redirect(url_for('index'))

def validateData(data):
    '''
    Validations of the input values.
    '''
    flag = False
    errors = {}
    
    for index, row in data.iterrows(): 
        
        if str(row['exec_ID'])=='nan':
            flag = True
            errors['exec_ID: '+str(row['exec_ID'])+' exec_ID'] = [' Must not be empty']

        if row['run_time']<0 or row[ 'run_time']== '':
            flag = True
            errors['exec_ID: '+str(row['exec_ID'])+' run_time'] = [' Must not be empty or a negative value']
            
        if row[ 'start_time']== '':
            flag = True
            errors['exec_ID: '+str(row['exec_ID'])+' start_time'] = [' Must not be empty']            

        if row['pass']<0 or row[ 'pass']== '':
            flag = True
            errors['exec_ID: '+str(row['exec_ID'])+' pass'] = [' Must not be empty or a negative value']            
            
    return flag, errors

def getData(data):      
    df = data[featureList] # Extract features the model was trained with.
    df['run_time'] = df['run_time'].apply(lambda x: str(x).replace(',','')).astype(float)
    df['exec_count'] = 1
    df['start_time'] = pd.to_datetime(df['start_time'], format = '%b %d_%Y $ %H:%M:%S.%f')
    df = df.set_index('start_time')
    df = df.resample('D').sum()
    df['run_time'] = df['run_time']/(1000*60*60)
    df.to_excel(os.getcwd()+'\\output_files\\df.xlsx')
    return df
    
def multivariate_time_window_2d(timeSeries, timeSteps, futureSteps):
  '''
  Featurizes time series into previous timeSteps 
  Returns a 2 dim X input. 
  '''
  X = []
  y = []
  for i in range(len(timeSeries)):
    if i+timeSteps+futureSteps > len(timeSeries):
      break
    X_l=[]
    for k in range(i, i+timeSteps):
      for f in range(3):
        X_l.append(timeSeries[k][f])    
    X.append(X_l)
    y_l = []
    for j in range(i+timeSteps, i+timeSteps+futureSteps):
      y_l.append(timeSeries[j][0])
    y.append(y_l)
  X = np.array(X)
  y = np.array(y)
  return X,y
  
def detect_anomaly(runtime):
  anomaly_threshold = 100
  if runtime > anomaly_threshold:
    return True
  else:
    return False

@app.route('/',methods = ['POST'])
def predict():
    
    uploadFiles()
    data = pd.read_excel(file_path)
    
    # Data Validation
    flag, errors = validateData(data)
    if(flag):
        return jsonify(errors)
    df = getData(data)
    X_te, y_te = multivariate_time_window_2d(df[-5:].values, timeSteps=5, futureSteps=0)
    
    Xq = scaler.transform(X_te)
    
    y_forecast = modelrf.predict(Xq)
    
    
    # Create a new dataframe with forecasted results.
    modelName = 'MultiVariate - MultiStep'
    dateIndex = pd.date_range(df.index[-1] + timedelta(days=1) , df.index[-1] + timedelta(days=7) , freq='D')
    dfF = pd.DataFrame(columns=['run_time(hrs)'])
    dfF.loc[df.index[-1]] = df['run_time'][-1]
    dfForecast = pd.DataFrame(data = y_forecast.reshape(7,1), columns = ['run_time(hrs)'], index= dateIndex)
    dfForecast = pd.concat([dfF, dfForecast], axis=0)

    # Set Anomaly Flags
    df['Anomaly_Flag'] = df['run_time'].apply(detect_anomaly)
    dfForecast['Anomaly_Flag'] = dfForecast['run_time(hrs)'].apply(detect_anomaly)
    result = dfForecast[1:8]
    #print('Forecast for next 7 days:\n',dfForecast[1:8])
    result.to_excel(os.getcwd()+'\\output_files\\result.xlsx')

    # Visualize
    ax = df['run_time'].plot(label='Historical', figsize=(20, 7),color='green')
    dfForecast['run_time(hrs)'].apply(lambda x : 0 if x<0 else x).plot(ax=ax, label='Forecast', color='blue')
    plt.scatter(x= df[df['Anomaly_Flag']==1].index, y= df[df['Anomaly_Flag']==True]['run_time'],color='red', label = 'Predicted Anomaly',linewidths=4)
    plt.scatter(x= dfForecast[dfForecast['Anomaly_Flag']==1].index, y= dfForecast[dfForecast['Anomaly_Flag']==1]['run_time(hrs)'],color='black', label = 'Forecasted Anomaly',linewidths=4)
    plt.xlabel('Date')
    plt.ylabel('Runtime(hrs)')
    plt.title(modelName+': Actual & Forecasted Runtimes')
    plt.legend()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'forecast.jpg'))    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast.jpg') 
    return render_template('results.html', user_image = full_filename, tables = [result.to_html(header=True)],  titles=[''])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
