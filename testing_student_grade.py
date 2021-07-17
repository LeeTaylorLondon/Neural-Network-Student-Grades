import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from testing_my_grades import build_model_a1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU


def load_data():
    return pd.read_csv('student_grades.csv')

def clean_data(df):
    """ This functions no values are missing """
    print(df.isnull().sum())

def data_preprocessing(df):
    """ Scale all data points except G3 """
    df_prescaled = df.copy()
    df_scaled = df.drop(['G3'], axis=1)
    df_scaled = scale(df_scaled)
    cols = df.columns.tolist()
    cols.remove('G3')
    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
    df_scaled = pd.concat([df_scaled, df['G3']], axis=1)
    df = df_scaled.copy()
    return df, df_prescaled

def split_data(df):
    x = df.loc[:, df.columns != 'G3']
    y = df.loc[:, 'G3']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    return x_train, x_test, y_train, y_test

def test_model(model, *data):
    if len(data) != 4: raise TypeError('Incorrect amount of data passed to test_model(...).')
    x_train, x_test, y_train, y_test = data
    train_pred = model.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Train RMSE: {:0.2f}".format(train_rmse))
    print("Test RMSE: {:0.2f}".format(test_rmse))


if __name__ == '__main__':
    gdf = load_data()
    # clean_data(df)
    # print(gdf)
    gdf, _ = data_preprocessing(gdf)
    x_train, x_test, y_train, y_test = split_data(gdf)
    model = build_model_a1(33)
    model.fit(x_train, y_train, epochs=20)
    test_model(model, x_train, x_test, y_train, y_test)
