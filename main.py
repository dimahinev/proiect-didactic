import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from datetime import timedelta
import os

# Заголовок приложения
st.title("Прогнозирование курса молдавского лея")

# Функция для загрузки данных
def load_default_file():
    default_file_path = 'usd_to_leu.csv'
    if os.path.exists(default_file_path):
        return pd.read_csv(default_file_path, sep=";")
    return None

# Загружаем файл по умолчанию или предлагаем пользователю выбрать файл
default_data = load_default_file()
uploaded_file = st.file_uploader("Загрузите файл с данными (по умолчанию используется usd_to_leu.csv)", type=["csv"])

# Используем данные из файла по умолчанию или загруженного файла
if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=";")
elif default_data is not None:
    st.info("Используется файл по умолчанию: usd_to_leu.csv")
    data = default_data
else:
    st.error("Файл usd_to_leu.csv не найден. Пожалуйста, загрузите файл.")
    st.stop()

# Обработка данных
data = data[data['Data'].str.match(r'^\d{2}\.\d{2}\.\d{4}$', na=False)]
data.columns = ["Date", "Exchange Rate"]
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
data['Exchange Rate'] = pd.to_numeric(data['Exchange Rate'], errors='coerce')

# Отображение данных
st.subheader("Данные")
st.write(data.head())

# Выбор: одна дата или диапазон
st.subheader("Выберите дату или диапазон для прогноза")
single_date_mode = st.radio(
    "Выберите режим:", 
    options=["Одна дата", "Диапазон дат"], 
    index=0
)

if single_date_mode == "Одна дата":
    forecast_date = st.date_input(
        "Выберите дату для прогноза", 
        min_value=data['Date'].iloc[-1] + timedelta(days=1)
    )
else:
    start_date = st.date_input(
        "Начало диапазона", 
        min_value=data['Date'].iloc[-1] + timedelta(days=1)
    )
    end_date = st.date_input(
        "Конец диапазона", 
        min_value=start_date
    )

# Раздел: ARIMA
st.subheader("Прогнозирование с использованием ARIMA")
train_size = int(len(data) * 0.8)
train, test = data['Exchange Rate'][:train_size], data['Exchange Rate'][train_size:]

p = st.slider("Параметр p", 0, 10, 5)
d = st.slider("Параметр d", 0, 2, 1)
q = st.slider("Параметр q", 0, 10, 0)

if st.button("Построить прогноз ARIMA"):
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    if single_date_mode == "Одна дата":
        # Прогноз для одной даты
        future_steps = (pd.Timestamp(forecast_date) - data['Date'].iloc[-1]).days
        forecast = model_fit.get_forecast(steps=future_steps)
        forecast_mean = forecast.predicted_mean

        st.markdown(
            f"## Прогноз курса на **{forecast_date}** - <span style='color: #84C9FF;'>{forecast_mean.iloc[-1]:.2f} MDL</span>",
            unsafe_allow_html=True
        )

        # Добавление точки прогноза на график
        future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=future_steps)
        forecast_data = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast_mean
        })

        combined_data = pd.concat([
            pd.DataFrame({"Date": data['Date'], "Value": data['Exchange Rate'], "Type": "Date de antrenament"}).iloc[:train_size],
            pd.DataFrame({"Date": data['Date'][train_size:], "Value": data['Exchange Rate'][train_size:], "Type": "Date reale (test)"}),
            pd.DataFrame({"Date": forecast_data['Date'], "Value": forecast_data['Forecast'], "Type": "Prognoza"}),
        ])

        st.line_chart(combined_data.pivot(index="Date", columns="Type", values="Value"))
    else:
        # Прогноз для диапазона дат
        future_steps = (pd.Timestamp(end_date) - data['Date'].iloc[-1]).days
        forecast = model_fit.get_forecast(steps=future_steps)
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()

        future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=future_steps)
        forecast_data = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast_mean,
            "Lower Bound": forecast_conf_int.iloc[:, 0],
            "Upper Bound": forecast_conf_int.iloc[:, 1]
        })

        combined_data = pd.concat([
            pd.DataFrame({"Date": data['Date'], "Value": data['Exchange Rate'], "Type": "Date de antrenament"}).iloc[:train_size],
            pd.DataFrame({"Date": data['Date'][train_size:], "Value": data['Exchange Rate'][train_size:], "Type": "Date reale (test)"}),
            pd.DataFrame({"Date": forecast_data['Date'], "Value": forecast_data['Forecast'], "Type": "Prognoza"}),
        ])

        st.line_chart(combined_data.pivot(index="Date", columns="Type", values="Value"))

# Раздел: LSTM
st.subheader("Прогнозирование с использованием LSTM")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Exchange Rate'].values.reshape(-1, 1))

train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

time_step = st.slider("Временной шаг для LSTM", 10, 60, 30)

def create_dataset(dataset, time_step=30):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

if st.button("Построить прогноз LSTM"):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    if single_date_mode == "Одна дата":
        # Прогноз для одной даты
        future_steps = (pd.Timestamp(forecast_date) - data['Date'].iloc[-1]).days
        last_data = test_data[-time_step:].reshape(1, time_step, 1)
        future_predictions = []
        for _ in range(future_steps):
            next_step = model.predict(last_data, verbose=0)
            future_predictions.append(next_step[0, 0])
            last_data = np.append(last_data[:, 1:, :], [[[next_step[0, 0]]]], axis=1)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        st.markdown(
            f"## Прогноз курса на **{forecast_date}** - <span style='color: #84C9FF;'>{future_predictions[-1, 0]:.2f} MDL</span>",
            unsafe_allow_html=True
        )

        # Добавление точки прогноза на график
        future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=future_steps)
        forecast_data = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_predictions.flatten()
        })

        combined_data = pd.concat([
            pd.DataFrame({"Date": data['Date'], "Value": data['Exchange Rate'], "Type": "Date de antrenament"}).iloc[:train_size],
            pd.DataFrame({"Date": data['Date'][train_size:], "Value": data['Exchange Rate'][train_size:], "Type": "Date reale (test)"}),
            pd.DataFrame({"Date": forecast_data['Date'], "Value": forecast_data['Forecast'], "Type": "Prognoza"}),
        ])

        st.line_chart(combined_data.pivot(index="Date", columns="Type", values="Value"))
    else:
        # Прогноз для диапазона дат
        future_steps = (pd.Timestamp(end_date) - data['Date'].iloc[-1]).days
        last_data = test_data[-time_step:].reshape(1, time_step, 1)
        future_predictions = []
        for _ in range(future_steps):
            next_step = model.predict(last_data, verbose=0)
            future_predictions.append(next_step[0, 0])
            last_data = np.append(last_data[:, 1:, :], [[[next_step[0, 0]]]], axis=1)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=future_steps)
        forecast_data = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_predictions.flatten()
        })

        combined_data = pd.concat([
            pd.DataFrame({"Date": data['Date'], "Value": data['Exchange Rate'], "Type": "Date de antrenament"}).iloc[:train_size],
            pd.DataFrame({"Date": data['Date'][train_size:], "Value": data['Exchange Rate'][train_size:], "Type": "Date reale (test)"}),
            pd.DataFrame({"Date": forecast_data['Date'], "Value": forecast_data['Forecast'], "Type": "Prognoza"}),
        ])

        st.line_chart(combined_data.pivot(index="Date", columns="Type", values="Value"))
