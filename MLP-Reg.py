'''
By h.alavi
'''
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd

inputdata=pd.read_csv(r'C:\Users\h.alavi\Documents\GitHub\APC-PRJ\input-data.csv')
outputdata=pd.read_csv(r'C:\Users\h.alavi\Documents\GitHub\APC-PRJ\output-data.csv')

X= inputdata.iloc[:,:].values
y=outputdata.iloc[:,10].values


y = y.reshape(-1,1)
#print(X)
#print(y)
print(X.shape,y.shape)

# standardizing
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_X.fit_transform(y)
#print(X_scaled)
#print(y_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled ,random_state=1, test_size=0.2)

#print(type(X_train))
#print(type(y_train))




reg = MLPRegressor(hidden_layer_sizes=(100,),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train.ravel())
y_pred=reg.predict(X_test)
y_pred_train = reg.predict(X_train)

print("The Predict Score is ", (r2_score(y_pred_train, y_train)))
print("The Train Score is: ", (r2_score(y_pred, y_test)))

