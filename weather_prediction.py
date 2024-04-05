import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title("Weather Prediction Web App")

precipitation = st.number_input("Enter the weather prediction: ")
Temperature_Maximum = st.number_input("Enter the maximum tenperature in Celsius: ")
Temperature_Minimum = st.number_input("Enter the minimum tenperature in Celsius: ")
Wind = st.number_input("Enter the wind speed: ")


df = pd.read_csv('seattle-weather.csv')
        # print(df)
        
        # Checking the data has null value or na values
print(df.info())
print(df.isna().count())
print(df.isnull().count())
        
        #Checking the min temp in the data to identify what is min temperature
print(df[df['temp_min'] == min(df['temp_min'])])
        
        #Checking the max temp in the data to identify what is max temperature
print(df[df['temp_max'] == max(df['temp_max'])])
        
print(df.describe())
        
# Histogram to know the count of maximum temperature
# plt.figure(figsize=(12,6))
# sns.histplot(data = df,x = df.temp_max,kde=True)
# plt.xlabel("Temperature Maximum")
# plt.ylabel("Count")
# plt.grid(True)
# plt.show()

# # # #Histogram to know the count of minimum temperature
# plt.figure(figsize=(12,6))
# sns.histplot(data = df,x = df.temp_min,kde=True)
# plt.xlabel("Temperature Minimum")
# plt.ylabel("Count")
# plt.grid(True)
# plt.show()

# date is not in proper format so let's convert it

df['date'] = pd.to_datetime(df['date'])
print(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

#Trying to know which year,month and which whether has max temperature
# g = sns.FacetGrid(df,col="year",hue="weather")
# g.map(sns.lineplot, 'month', 'temp_max').add_legend()
# # g.map(sns.scatterplot,'month', 'temp_max').add_legend()
# g.set_axis_labels('Month', 'Max Temperature (°C)')
# plt.show()

# # #Trying to know which month has max temperature
# g = sns.FacetGrid(df,col="year")
# g.map(sns.lineplot, 'month', 'temp_max').add_legend()
# g.set_axis_labels('Month', 'Max Temperature (°C)')
# plt.show()

# # #Trial
# g.map(sns.scatterplot,'month', 'temp_max')
# g.set_axis_labels('Month', 'Max Temperature (°C)')
# plt.show()

# sns.lineplot(data=df,x=df['month'],y=df['temp_max'],hue=df['year'])
# plt.xlabel("Month")
# plt.ylabel("Max-Temperature")
# plt.show()


# sns.lineplot(data=df,x=df['month'],y=df['temp_min'],hue=df['year'])
# plt.xlabel("Month")
# plt.ylabel("Max-Temperature")
# plt.show()

# sns.scatterplot(data=df,x=df['month'],y=df['precipitation'],hue=df['weather'])
# plt.xlabel("Month")
# plt.ylabel("precipitation")
# plt.show()

# sns.scatterplot(data=df,x=df['month'],y=df['precipitation'],hue=df['weather'])
# plt.xlabel("Month")
# plt.ylabel("precipitation")
# plt.show()

# sns.countplot(data = df,x=df['weather'])
# plt.xlabel("Weather")
# plt.ylabel("Count")
# plt.show()

# plt.pie(x=df['weather'].value_counts(),autopct='%1.1f',labels=df['weather'].value_counts().index)
# plt.title('Distribution of Weather Types')
# plt.show()

# There is no neccessity of month and year
#0-> index,1->column
df.drop('month',axis=1,inplace=True)
df.drop('year',axis=1,inplace=True)
# print(df)
        
le = LabelEncoder()
        # 4-> sun , 1->fog , 2-> rain , 3-> snow 4->drizzle
df['weather'] = le.fit_transform(df['weather'])
print(df['weather'])
        
x = df[['precipitation','temp_max','temp_min','wind']]
# print(x)
y = df['weather']
X_train,X_test,Y_train,y_test = train_test_split(x,y,random_state=30,test_size=0.2)
# print(X_train)# contains data in 'precipitation','temp_min','temp_max','wind' format
# print(Y_train)# contains data in weather format
# print(X_test) # contains data in 'precipitation','temp_min','temp_max','wind' format
# print(y_test) # contains data in weather format   
d = GaussianNB()
# print(d)
d.fit(X_test.values,y_test.values)

y_pred = d.predict(X_test.values)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
score = d.score(X_test,y_test)
print(score)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
example_prediction = d.predict([[precipitation,Temperature_Maximum,Temperature_Minimum,Wind]])
Predicted_value = le.inverse_transform(example_prediction)[0]
print(Predicted_value)

if(st.button('Weather_Predict')):
    st.success("The Weather is: "+Predicted_value)
    st.info(conf_matrix)
    st.info(classification_rep)
    
    
st.title("EDA")
st.image("static/Figure_1.png")
st.image("static/Figure_2.png")
st.image("static/Figure_3.png")
st.image("static/Figure_4.png")
st.image("static/Figure_5.png")
st.image("static/Figure_6.png")
st.image("static/Figure_7.png")
st.image("static/Figure_8.png")
st.image("static/Figure_9.png")
st.image("static/Figure_10.png")