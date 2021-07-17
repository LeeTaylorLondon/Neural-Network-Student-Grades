# Author: Lee Taylor
# Personal project to check what I've learned from my previous
# ML & NN projects on https://github.com/LeeTaylorNewcastle/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential


# --[1st Year Grades (bad)]--
report = [65, 84, 74]
coding = [87.5, 94, 65, 55, 57, 55, 60, 100, 84, 80]
exam = [90, 90, 90]
output = [76.1]

# --[2nd Year Grades]--
report2 = [100, 100, 83.5, 85]
coding2 = [93, 94, 75]
exam2 = [95, 90]
output2 = [87.3]


def preprocess_report(rep, rep2):
    """ Processes lists containing report grades """
    rv = np.asarray([rep], dtype="float32")
    rv2 = np.asarray([rep2], dtype="float32")
    return rv, rv2

def preprocess_all():
    """ Processes all lists of grades by concatenating them """
    return np.asarray([np.concatenate((report, coding, exam, output))], dtype="float32")

def build_model_r3(input_shape, output_shape):
    """ Model for predicting report grades. Note the activation
    is LeakyReLU because ReLU would pre-emptively drop units
    leading to incorrect predictions of [0. 0. ...].
     The 'r' in 'r3' denotes 'report', '3' denotes version. """
    model = Sequential()
    model.add(Dense(units=32, input_dim=input_shape))
    model.add(LeakyReLU())
    model.add(Dense(units=16))
    model.add(LeakyReLU())
    model.add(Dense(units=8))
    model.add(LeakyReLU())
    model.add(Dense(units=output_shape))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def build_model_a1(input_shape, output_shape=1):
    model = Sequential()
    model.add(Dense(units=128, input_dim=input_shape))
    model.add(LeakyReLU())
    model.add(Dense(units=64))
    model.add(LeakyReLU())
    model.add(Dense(units=64))
    model.add(LeakyReLU())
    model.add(Dense(units=32))
    model.add(LeakyReLU())
    model.add(Dense(units=1))
    model.add(LeakyReLU())
    model.add(Dense(units=output_shape))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def model_predict(model, inp, ans):
    """ Passed model outputs it's prediction and extra data
     such as the input passed, and the output rounded """
    print(f"\nModel.predict({inp})")
    predict = model.predict(inp)
    # Out data
    print(f"{predict} Model prediction\n{ans} Answer")
    for i, g in enumerate(predict[0]): predict[0][i] = round(g, 1)
    print(f"{predict} Model prediction rounded")
    print(model.evaluate(inp, ans))

def test_report(verbose=True):
    """ Preprocess report grades, build model for reports, fit model,
    query trained/fitted model on input and output it's guess """
    debug = True
    r, r2, = preprocess_report(report, report2)
    report_model = build_model_r3(3, 4)
    if debug: print(f"report shapes: {r.shape}, {r2.shape}")
    # Fit model and predict
    report_model.fit(r, r2, epochs=200,verbose=verbose)
    if verbose: model_predict(report_model, r, r2)

def test_all(debug=False):
    """ Preprocess all grades, build model suited for all grades, fit model,
    query trained/fitted model on input and output it's guess """
    inp = preprocess_all()
    out = np.asarray([output2[0]], dtype="float32")
    if debug:
        print(inp, out)
        print(inp.shape, out.shape)
        print(inp.shape[0])
    model = build_model_a1(inp.shape[1], 1)
    model.fit(inp, out, epochs=100)
    model_predict(model, inp, out)


if __name__ == '__main__':
    # test_report(verbose=True)
    # test_all(debug=False)

    grades = preprocess_all()
    grades = np.reshape(grades, (17, 1))
    gdf = pd.DataFrame(grades) # grades data frame
    print(gdf)
    print(dir(gdf))
    gdf.plot(kind='bar', ylim=(0, 120), colormap='terrain')
    # plt.plot(gdf, )
    plt.ylabel('Out of 100%')
    plt.xlabel('Grade Percentages')
    plt.show()

    # grades = preprocess_all()
    # grades.sort()
    # grades = np.reshape(grades, (17, 1))
    # gdf = pd.DataFrame(grades)  # grades data frame
    # gdf.plot(kind='bar', ylim=(0, 115))
    # # plt.plot(gdf, )
    # plt.title('Grades sorted plotted')
    # plt.ylabel('Out of 100%')
    # plt.xlabel('Grade Percentages')
    # plt.show()

    pass