import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn  import linear_model
from sklearn.metrics import r2_score

ds = pd.read_csv('FuelConsumption.csv')
ds.head(n=5)

f_ds = ds[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 
f_ds.head()



viz = f_ds
viz.hist()
plt.show()

X = f_ds['ENGINESIZE']
Y = f_ds['CO2EMISSIONS']
plt.title("----Emission----")
plt.scatter(X,Y,color='green')
plt.xlabel("Engine size ")
plt.xlabel("CO2 emission ")
plt.show()

msk = np.random.rand(len(ds)) < 0.8
train = f_ds[msk]
test = f_ds[~msk]




plt.scatter(train['ENGINESIZE'],train['CO2EMISSIONS'],color='green')
plt.show()



lin_model = linear_model.LinearRegression()
model_X = np.asanyarray(train[['ENGINESIZE']])                        
model_Y = np.asanyarray(train[['CO2EMISSIONS']])                        
lin_model.fit(model_X,model_Y)

print("Co-efficient : ",lin_model.coef_)
print("Intercept    : ",lin_model.intercept_)


X=train['ENGINESIZE']
Y=train['CO2EMISSIONS']
plt.scatter(X,Y,color='red')
plt.plot(model_X,lin_model.coef_[0][0]*model_X + lin_model.intercept_[0],color='blue')
plt.show()





evalu_X = np.asanyarray(test[['ENGINESIZE']])
evalu_Y = np.asanyarray(test[['CO2EMISSIONS']])
evalu_answer = lin_model.predict(evalu_Y)



Mean_absolute_error = np.mean(np.absolute(evalu_answer - evalu_Y))

print( "Mean_absolute_error  : %.2f "%(Mean_absolute_error) )

Residual_sum_of_squares = np.mean((evalu_answer - evalu_Y) ** 2)

print("Residual_sum_of_squares : %.2f " %(Residual_sum_of_squares) )

R2_score =r2_score(evalu_answer , evalu_X)

print("R2_score : %.2f " % (R2_score))


print("----------------------------------------------- Thank you ---------------------------------------")
