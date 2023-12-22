import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from google.colab import files


# opening the file with google co-lab
#-----------------------------------------------------------------------------------------------------------------------------------------

### Prompt to upload a file ###
#uploaded = files.upload()

# Print the uploaded file details ###
#for filename in uploaded.keys():
    #print(f'Uploaded file: {filename}, {len(uploaded[filename])} bytes')
#-----------------------------------------------------------------------------------------------------------------------------------------


### change this path after BostonHousing.csv is downloaded ###
housing_dataset  = pd.read_csv('your path to BostonHousing.csv')



### finding the outliers ###
z_scores = ((housing_dataset - housing_dataset.mean()) / housing_dataset.std()).abs()
outliers = z_scores > 3

### Exclude outliers and create a new dataset ###
filtered_data = housing_dataset[~outliers.any(axis=1)]



### Correlation ###
plt.rcParams["figure.figsize"] = [8,6]
corr = filtered_data.corr()
print(corr)
sns.heatmap(corr)

### features ###
X = filtered_data.drop(["MEDV"], axis = 1)
### labels ###
y = filtered_data.filter(["MEDV"], axis = 1)


### Splits data into 80% training and 20% testing ###
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

house_predictor = LinearRegression()

### trains the algorithm with our data ###
house_predictor.fit(X_train, y_train)

### predicts testing data ###
y_pred = house_predictor.predict(X_test)

### metrics ###
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

### compares predicted values with their true values ###
comparison_df = pd.DataFrame({'Actual': y_test.values.tolist(), 'Predicted': y_pred.tolist()})
print(comparison_df)

### metric that the algorithm uses to predict ###
print(house_predictor.coef_)

### shape of single point ###
#X_test.values[1].shape

### Reshapes single point ###
single_point = X_test.values[1].reshape(1,-1)

print('predicted', house_predictor.predict(single_point))
print('actual', y_test.values[1])

def housing_price(feature):
  fet_pred = house_predictor.predict(feature)
  return fet_pred
