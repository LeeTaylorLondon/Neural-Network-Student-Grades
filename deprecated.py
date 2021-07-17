import numpy as np
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential

def preprocess_report(rep, rep2, avg=True):
    if avg: rep.append(sum(rep) / len(rep))
    rv = np.asarray([rep], dtype="float32")
    rv2 = np.asarray([rep2], dtype="float32")
    return rv, rv2

def build_report_model_1():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=4))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def build_report_model_2():
    model = Sequential()
    model.add(Dense(units=32, input_dim=4))
    model.add(LeakyReLU())
    model.add(Dense(units=16))
    model.add(LeakyReLU())
    model.add(Dense(units=8))
    model.add(LeakyReLU())
    model.add(Dense(units=4))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model


def test_equal_io():
    debug = True
    r, r2, = preprocess_report(report, report2)
    report_model = build_report_model_2()
    # Debug
    if debug:
        print(f"report shapes: {r.shape}, {r2.shape}")
    # Debug
    report_model.fit(r, r2, epochs=200)
    print(f"Model.predict({r})")
    predict = report_model.predict(r)
    # Out data
    print()
    print(predict)
    print(r2)
    for i,g in enumerate(predict[0]): predict[0][i] = round(g, 1)
    print(predict)