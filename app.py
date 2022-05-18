import numpy as np
import pandas as pd
import math
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import joblib
import pickle
import _sqlite3
conn = _sqlite3.connect('user.db')
c = conn.cursor()

# create a table
def userTable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

# add users data to the database
def add_user(username, password):
    c.execute('INSERT INTO userstable(username, password) VALUES(?,?)',(username, password))
    conn.commit()

# get data from login users
def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username=? AND password=?',(username, password))
    data = c.fetchall()
    return data

# views all users in the database
def view_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def main():

    st.title('Stock Hub ~ Stock Price Prediction Dashboard')
    menu = ['Home','Login','Register']
    submenu = ['plot','prediction']

    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader('Home')

    #     st.text('In this machine learning application, I have used the historical stock price data for '
    # 'different tech companies to forecast their future prices.')
    elif choice == 'Login':
        st.subheader('Login section')
        username = st.sidebar.text_input('User Name')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.checkbox('Login'):
            userTable()
            # authenticate user
            user = login_user(username, password)
            if user:
                st.success(f'Welcome {username}')
            else:
                st.warning('wrong username/password')

    elif choice == 'Register':
        st.subheader('Register an account')
        new_user = st.text_input('Username')
        new_password = st.text_input('password', type='password')
        if st.button('Register'):
            userTable()
            add_user(new_user, new_password)
            st.success('You have succesfully created an account')
            st.info('navigate to the login page')


if __name__ == '__main__':
    main()

# cover image
image_path = ('ima.jpg')
st.image(image_path,use_column_width=True)

st.header('Welcome to Stock Hub! ')

st.markdown(
    'In this machine learning application, I have used the historical stock price data for '
    'different tech companies to forecast their future prices.'
    )

st.markdown(
    'I have used the keras framework from Tensorflow to build an LSTM model.'

)

st.write("""
A web application that retrives data from an API source and integrated Machine learning to forcast stock prices for tech companies
* Python Libraries: pandas, numpy, matplotlib, streamlit, seaborn, yfinance
* Data source: yahoo finance
""")



ticker = ['FB','AMZN', 'GOOGL', 'TSLA', 'AAPL', 'NFLX']
def get_user_input():
    # startDate = st.text_input('Enter starting date as YYYY-MM-DD', '2000-01-10')
    # 'Enter the start date: ', startDate
    # endDate = st.text_input('Enter ending date as YYYY-MM-DD', '2000-01-20')
    # 'Enter the end date: ', endDate
    selected_stock = st.selectbox('Enter company stock symbol', ticker)
    return selected_stock

# get user input
stock = get_user_input()




# function to get data
startDate = datetime.datetime(2017,3,1)
endDate = datetime.datetime(2022,3,1)
# @st.cache
def load_data(ticker, start, end):
   data = yf.download(ticker, start=start, end=end)
   data.reset_index(inplace=True)
   return data

# get the data
data_load = st.text('load data...')
data = load_data(stock, startDate, endDate)
data_load.text('loading...done!')

clicked = st.button('click here')



# create checkbox
if st.checkbox(f'Stock Market data from {startDate} to {endDate} for {stock}'):
    st.subheader('Raw data')
    st.write(data)

# function to visualise the data
def plot_raw_data(data):
    fig = plt.figure(figsize=(12,6))
    plt.xlabel('Year')
    plt.title("{} Closing stock price".format(stock), fontdict={'fontsize':18})
    plt.ylabel('Closing stock price USD ($)', fontdict={'fontsize':18})
    sns.lineplot('Date','Close', data=data)
    st.pyplot(fig)
    # st.line_chart(data.Close)

plot_raw_data(data)

st.write(
    """
    ## Stock Open Price
    """
)
st.line_chart(data.Open)

st.write("""
## Stock Low price 
""")

fig = plt.figure(figsize=(12,6))
plt.plot(data['Low'])
plt.title(f'low stock price for {stock}', fontsize=18)
st.pyplot(fig)

st.write("""
## volume Price 
""")
st.line_chart(data.Volume)




# show data description
st.subheader('Descriptive characteristics about the dataset')
describe = data.describe()
st.write(describe)


# Forcasting machine learning

st.header('Stock prediction model')
st.write(f"""
    ### Selected company: {stock}
""")

# model_type = ['linear regression', 'LSTM']
#
# def check_model():
#     selected_model = st.selectbox('Pick a model', model_type)
#     return selected_model
#
# model = check_model()

#*************Fitting the model**************************
def linear_model(data):
    data.set_index('Date', inplace=True)
    # drop columns
    cols = ['Volume', 'Adj Close']
    data.drop(cols, axis=1, inplace=True)

    # create independent and target features
    X = data[['Open', 'Low', 'High']]
    y = data['Close']

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # fit the linear model
    model = LinearRegression()
    model.fit(x_train, y_train)
    # print(model.coef_)
    # # Testing
    predictions = model.predict(x_test)
    new_df = pd.DataFrame({'Actual_price': y_test, 'Predicted_price': predictions})
    new_df.reset_index(inplace=True)

    # calculate error
    RMSE = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
    MSE = round(mean_squared_error(y_test, predictions), 2)
    MAE = round(mean_absolute_error(y_test, predictions), 2)

    # #  visualise the predictions
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(x='Date', y='Actual_price', data=new_df, color='blue')
    sns.lineplot(x='Date', y='Predicted_price', data=new_df, color='red')
    sns.set_style(style='dark')
    plt.title("Linear Regression Model prediction", fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.ylabel('Closing stock price in USD ($)', fontsize=15)
    plt.xlabel('Year', fontsize=15)
    plt.legend(labels=['Actual_price', 'Predicted_price'], loc='upper left', fontsize=15)
    # plt.show()
    st.pyplot(fig)
    # st.write(f"""
    #             The mean square error of prediction is {MSE},
    #             mean absolute error of prediction is {MAE}, and the
    #             the root mean square error of prediction is {RMSE}
    #             """
    #          )
    return new_df

predicted = linear_model(data)
st.subheader('result from the linear regression model')
st.write(predicted)

def LSTM_model(df):

    df = data.filter(['Close'])
    # convert the dataframe to a numpy array
    dataset = df.values
    print(dataset)
    # split data into training and testing set
    training_data_size = math.ceil(len(dataset) * 0.8)
    # training_data_size

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # creating data with 30 timestamp and 1 output
    # store data from 30 days before predicting the next stock price
    train_data = scaled_data[0:training_data_size, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        # if i <= 60:
        #     print(x_train)
        #     print(y_train)
        #
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #reshape data before building the model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Build LSTM

    classifier = Sequential()
    # first LSTM layer
    classifier.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    classifier.add(Dropout(0.2))

    # second LSTM layer
    classifier.add(LSTM(50, return_sequences=True))
    classifier.add(Dropout(0.2))

    # Third LSTM layer
    classifier.add(LSTM(50))
    classifier.add(Dropout(0.2))

    # output layer
    classifier.add(Dense(1))

    # compile the model
    classifier.compile(optimizer='adam', loss='mean_squared_error')

    # fit the data
    classifier.fit(x_train, y_train, epochs=15, batch_size=64)


    # testing
    # create test data
    test_data = scaled_data[training_data_size - 60:, :]
    x_test = []
    y_test = dataset[training_data_size:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # convert list to numpy arrays
    x_test = np.array(x_test)

    # reshape the test data
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # testing predictions
    predictions = classifier.predict(x_test)
    # transforming to get the origin data before scaling
    predictions = scaler.inverse_transform(predictions)

    # calculate error
    RMSE = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
    MSE = round(mean_squared_error(y_test, predictions), 2)
    MAE = round(mean_absolute_error(y_test, predictions), 2)

    # make plot
    train = data[:training_data_size]
    valid = data[training_data_size:]
    valid['predictions'] = predictions
    fig = plt.figure(figsize=(12, 8))
    plt.title('LSTM Forcasting Model')
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Closing stock price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'predictions']])
    plt.legend(['Actual', "valid", 'predicted'], loc='upper left')
    # plt.show()
    st.pyplot(fig)

    # st.write(f"""
    #         The mean square error of prediction is {MSE},
    #         mean absolute error of prediction is {MAE}, and the
    #         the root mean square error of prediction is {RMSE}
    #         """
    #          )
    return valid

predicted = LSTM_model(data)
st.write(predicted)




# st.header('Future stock prices prediction')
# st.write(f"""
#     ### Selected company: {stock}
# """)
#*************Model forecast****************
# def Linear_model(data):
#     df = data[['Close']]
#     # number of days to forecast in the future
#     forecast_days = 30
#     # create new  column for the prices after n days
#     df['prediction'] = df[['Close']].shift(-forecast_days)
#     # new_df = data[['Close', 'prediction']]
#
#     # create the feature dataset and convert to numpy array and remove the last x days
#     X = np.array(df.drop(['prediction'], 1))[:-forecast_days]
#     # create the target and convert to numpy array and get all target values except the last x rows
#     y = np.array(df['prediction'])[:-forecast_days]
#
#     # feature scaling - normalization
#     # scaler = MinMaxScaler(feature_range=(0,1))
#     # X = scaler.fit_transform(X)
#
#     # splitting the dataset into train and test
#     # training 80% and testing 20%
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#
#     # model building
#     linearModel = LinearRegression()
#     linearModel.fit(x_train, y_train)
#
#     # get the last x rows of the feature dataset
#     x_future = df.drop(['prediction'], 1)[:-forecast_days]
#     x_future = x_future.tail(forecast_days)
#     x_future = np.array(x_future)
#     # x_future
#     # testing
#     predictions = linearModel.predict(x_future)
#
#     prediction = predictions
#     valid = df[X.shape[0]:]
#     valid['prediction'] = prediction
#     fig = plt.figure(figsize=(16, 8))
#     plt.title("Linear Model Forecast")
#     plt.ylabel('Closing Price USD ($)')
#     plt.xlabel('Year')
#     plt.plot(df['Close'])
#     plt.plot(valid[['Close', 'prediction']])
#     plt.legend(['Actual', 'Valid', 'Predicted'])
#     st.pyplot(fig)
#
#     linear_error = np.sqrt(mean_squared_error(x_future, predictions))
#     error = 'The predicted data has RMSE value is'.format(linear_error)
#     st.write(error)
#     return valid

#
# result = Linear_model(data)
# st.subheader('The result of the forcast')
# st.write(result)















