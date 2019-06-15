import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import LabelEncoder



def loadCSV():

    columns = ['Month','Advertising','Sales']
    sales = pd.read_csv('advertising-and-sales-data-36-co.csv', header=0, names=columns)
    df = pd.DataFrame(sales)
    date_split = df['Month'].str.split('-').str
    df['Year'], df['Month'] = date_split 

    return  df[['Year', 'Month', 'Advertising', 'Sales']]

def dataPreProcessing(df):
   
    labelencoder = LabelEncoder()
    month_encoded = labelencoder.fit_transform(df['Month'])
    df['Month'] = month_encoded

    df['Month'] = df['Month']/100
    df['Advertising'] = df['Advertising']/100
    df['Sales'] = df['Sales']/100


    #training
    #f.append({'Animal':'mouse', 'Color':'black'}, ignore_index=True)
    return df.drop(['Year'], axis=1)


def dataset(df):

    window = 2

    qt_atributos = len(df.columns)
    mat_dados = df.as_matrix() #converter dataframe para matriz (lista com lista de cada registo)
    tam_sequencia = window + 1

    res = []
    for i in range(len(mat_dados) - tam_sequencia): #numero de registos - tamanho da sequencia
        res.append(mat_dados[i: i + tam_sequencia])

    res = np.array(res) #dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)
    qt_casos_treino = 24 #90% passam a ser casos de treino


    train = res[:qt_casos_treino, :]
    x_train = train[:, :-1] #menos um registo pois o ultimo registo é o registo a seguir à janela
    y_train = train[:, -1][:,-1] #para ir buscar o último atributo para a lista dos labels
    x_test = res[qt_casos_treino:, :-1]
    y_test = res[qt_casos_treino:, -1][:,-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))


    return [x_train, y_train, x_test, y_test]


def model():
    window = 2

    model = Sequential()
    model.add(LSTM(100, input_shape=(window, 3), return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(100, input_shape=(window, 3), return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(100, input_shape=(window, 3), return_sequences=False))
    model.add(Dropout(0.4))

    model.add(Dense(10, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(10, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model


def plot_prediction(y_test,predic):
    diff=[]
    racio=[]
    for i in range(len(y_test)): #para imprimir tabela de previsoes
        racio.append( (y_test[i]/predic[i])-1)
        diff.append( abs(y_test[i]- predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i],predic[i], diff[i], racio[i]))
    plt.plot(y_test,color='blue', label='y_test')
    plt.plot(predic,color='red', label='prediction') #este deu uma linha em branco
    plt.plot(diff,color='green', label='diff')
    plt.plot(racio,color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()



x_train, y_train, x_test, y_test,  = dataset(dataPreProcessing(loadCSV()))

lstm = model()

lstm.fit(x_train, y_train, epochs=2500, validation_split=0.1, batch_size=7, verbose=1)

trainScore = lstm.evaluate(x_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = lstm.evaluate(x_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

y_pred = np.squeeze(np.asarray(lstm.predict(x_test)))

plot_prediction(y_test,y_pred)
