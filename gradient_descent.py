# Juan David Torres
# A01702686
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

# Error storage
ERRORS= [];  


def hypothesis(params, sample):
	acum = np.dot(np.array(params),np.array(sample))
	return acum


def error_calculations(params, samples,y):
	global ERRORS
	error_acum =0

	for i in range(len(samples)):
		hyp = hypothesis(params,samples[i])
		error=hyp-y[i]
		error_acum=+error**2 
	mean_error_param=error_acum/len(samples)
	ERRORS.append(mean_error_param)

def gradient_descent(params, samples, y, alfa):
	temp = list(params)
	for j in range(len(params)):
		acum =0; error_acum=0
		for i in range(len(samples)):
			error = hypothesis(params,samples[i]) - y[i]
			acum = acum + error*samples[i][j]  
		temp[j] = params[j] - alfa*(1/len(samples))*acum  
	return temp



# Main
columns = ["No", "Transaction Date", "House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude","Price UPA"]
df = pd.read_csv('./RealEstate.csv',names = columns)

print(df.head())
cleaned_df = df[["House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude","Price UPA"]]

print(cleaned_df.head())
params = [0,0,0,0,0,0]
samples = cleaned_df[["House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude"]] ##.values.tolist()
y = cleaned_df["Price UPA"].values.tolist()

# Adding the 1 to each of the instances
samples.insert(0,'Beta',1)


alfa = 0.01  

#Scaling 
samples = samples.div(1000)

print ("scaled samples:")
print (samples)

samples = samples.values.tolist()

number_epochs = 20000
for i in range(number_epochs):  
	oldparams = list(params)
	params=gradient_descent(params, samples,y,alfa)	
	error_calculations(params, samples, y)  

print("Last parameters",params)

	
print("Mean squared error", ERRORS[-1])
# Plot of the mse
plt.plot(ERRORS)
plt.show()

# User Input Prediction
print("Enter the following in order separated by spaces")
print("House Age, Distance to nearest station, Number of Convinience Stores, Latitude and Longitude")
query = list(map(float,input().split()))

# For the beta
query.insert(0,1)
# Scaling
query = np.asarray(query)/1000

# Result of query
result = hypothesis(params,query)
print("Result: ",result)