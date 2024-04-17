# ====================================================================================================================================== #
#                                                   LR - MSE - GD - MMN - EG - LG - DP - C                                               #
# ====================================================================================================================================== #
# Author: Miguel Marines

# Project:
# 1. Linear Reggresion
# 2. Mean Square Error
# 3. Gradient Descent
# 4. Min-Max Normalization
# 5. Error and Loss Graph
# 6. Data Set Processing
# 7. Computations

# ====================================================================================================================================== #


# Library
import numpy as np
__error__ = []





# ====================================================================================================================================== #
#                                                          HYPOTESIS FUNCTION                                                            #
# ====================================================================================================================================== #
# Hypothesis Function: y = θ₁x₁ + θ₂x₂ + θ₃x₃ ... + b
# Hypothesis Function: y = m₁x₁ + m₂x₂ + m₃x₃ ... + b

# An extra x₀ (atribute) with value of 1 for the b (bias).

# Hypothesis Function: y = θ₀x₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ...
# Hypothesis Function: y = m₀X₀ + m₁x₁ + m₂x₂ + m₃x₃ ...

# parameters -> Parameter θ or m.
# x_features -> Feature x.

def hypothesis_function(parameters, x_features):
	summation = 0                                                    	 	# Acumulates the computations.
	for i in range(len(parameters)):                              			# Executes one case from the data set (1 Row).
		summation = summation + (parameters[i] * x_features[i])   			# Computation of the hypothesis function.
	return summation                                                 	 	# Returns the y (result value), obtained by the hypothesis function.

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                      MEAN SQUARE ERROR FUNCTION                                                        #
# ====================================================================================================================================== #
# Mean Square Error Function: MSE = 1/n * Σ(X₁ - Y₂)²

# parameters -> Parameter θ or m.
# x_features -> Feature x.
# y_results -> Expected results.

def mean_square_error_function(parameters, x_features, y_results):
    
	global __error__
    
	# Variable to store the summation of differences.
	acumulated_error = 0
	for i in range(len(x_features)):

		# HYPOTHESIS FUNCTION
		y_hypothesis = hypothesis_function(parameters, x_features[i])

		# Prints the computated result and the real result.
		print( "Computed Y:  %f  Real Y: %f " % (y_hypothesis,  y_results[i]))
		
		# MSE per case.
		# Mean square error function computation with: MSE = 1/n * Σ(X₁ - Y₂)²
		error = y_hypothesis - y_results[i]
		acumulated_error = error ** 2
	
	mean_square_error = acumulated_error / len(x_features)

	# Returns the mean square error value.
	__error__.append(mean_square_error)

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                       GRADIENT DESCENT FUNCTION                                                        #
# ====================================================================================================================================== #
# Gradient Descent Function: θj = θj - α/m Σ[(hθ(Xi) - Y)Xi]

# parameters -> Parameter θ or m.
# x_features -> Data x inputs of features.
# y_results -> List containing the corresponding real result for each sample.
# alfa -> Learning rate.

def gradient_descent_function(parameters, x_features, y_results, alfa):
	
	# Creates list of the lenght of the parameters.
	gradient_descent = list(parameters)
	
	# Gradient descent computation with: θj = θj - α/m Σ[(hθ(Xi) - Y)Xi]
	for i in range(len(parameters)):
		
		summation = 0
		
		for j in range(len(x_features)):
			error = hypothesis_function(parameters, x_features[j]) - y_results[j]
			summation = summation + (error * x_features[j][i])
		gradient_descent[i] = parameters[i] - (alfa * (1/len(x_features)) * summation)

	# Returns the errors from the thetas and the bias.
	return gradient_descent

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                      		   SCALING DATA SET                                                          #
# ====================================================================================================================================== #
# SCALING -> Normalization or Standarization.

# Normalizes the sample values so that the gradient descent can converge.
# Normalization maps the data in the range 0 to 1.

def scaling_function(x_features):

	acum = 0

	# Get same atributes from all cases and convert them to list.
	# (θ₁, θ₁, θ₁, θ₁, θ₁)
	# (θ₂, θ₂, θ₂, θ₂, θ₂)
	# (θ₃, θ₃, θ₃, θ₃, θ₃)
	x_features = np.asarray(x_features).T.tolist()

	# Cycles to go case by case and atrtibute by atribute to normalize.
	# Doesn't do the first parameter θ₀, because is the bias and it is always 1.
	for i in range(1, len(x_features)):
		
		for j in range(len(x_features[i])):
			# Get addition af all the values of the same atributes.
			# (θ₁ + θ₁ + θ₁ + θ₁ + θ₁)
			# (θ₂ + θ₂ + θ₂ + θ₂ + θ₂)
			# (θ₃ + θ₃ + θ₃ + θ₃ + θ₃)
			acum =+ x_features[i][j]

		# Minimum value of the same kind of atribute.
		min_val = min(x_features[i])

		# Maximum value of the same kind of atribute.
		max_val = max(x_features[i])

		# Minimum - Maximum Scaling = (X - Xmin) / (Xmax - Xmin)
		for j in range(len(x_features[i])):
			x_features[i][j] = (x_features[i][j] - min_val) / (max_val - min_val)

	return np.asarray(x_features).T.tolist()

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                         1. Process Data Set                                                            #
# ====================================================================================================================================== #
# Import x's (atributes) from the data set.
x_features = np.loadtxt('/Users/.../...', dtype = int, delimiter = ',')

# Delete target from the atributes (delete y from x's).
x_features = np.delete(x_features, 3, axis = 1)

# Insert an extra x₀ (atribute) with value of 1 for the b (bias).
x_features = np.insert(x_features, 0, 1, axis = 1)

# Hypothesis Function: y = θ₁x₁ + θ₂x₂ + θ₃x₃ ... + b
# New hypothesis function. y = θ₀x₀ + θ₁x₁ + θ₂x₂ + θ₃x₃

# Convert to integer type the atributes.
x_features = x_features.astype('int')

# Print x's (features).
#print(x_features)

# Import y (target) from the data set.
y_results = np.loadtxt("/Users/.../...", usecols = 3, dtype = int, delimiter = ",")
# Print y (target).
#print(y_results)

# Original data set.
print ("Original Samples:")
print (x_features)

# Scale the data set with minimum - maximum normalization.
x_features = scaling_function(x_features)

# Scaled data set.
print ("Scaled Samples:")
print (x_features)

# Hypothesis Function: y = θ₁x₁ + θ₂x₂ + θ₃x₃ ... + b
# New hypothesis function. y = θ₀x₀ + θ₁x₁ + θ₂x₂ + θ₃x₃

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#	                                                         2. Computation                                                              #
# ====================================================================================================================================== #
# Learning rate
# alfa = 0.0003  	
alfa = 0.003

# Initial θs (patameters).
parameters = [0,0,0,0]

# Epoch - Training Iterations.
epoch = 0

# Cycle to run the functions (LR, MSE, DG) until the parameters remain the same (minimum error) or the epoch 
# (training iterations) are reacehd.
while True:
	
	# Old parameters to work with.
	old_parameters = list(parameters)
	#print(parameters)
	
	# DESCENDING GRADIENT
	parameters = gradient_descent_function(parameters, x_features, y_results, alfa)	
	
	# MEAN SQUARE ERROR (Shows errors, Not used in calculations.)
	mean_square_error_function(parameters, x_features, y_results)
	
	# Print the new calculated parameters.
	#print(parameters)

	# Addition of the learning iteration.
	epoch = epoch + 1
	# print("Epoch: ", epoch)

	# When the the parameters remain the same (minimum error) or the epoch (training iterations) are reacehd, print the result.
	if(old_parameters == parameters or epoch == 10000):
		#print("Samples:")
		#print(x_features)
		#print("Final Params:")
		#print(parameters)		
		break
# ====================================================================================================================================== #





# ====================================================================================================================================== #
#	                                                             3. Graph                                                              	 #
# ====================================================================================================================================== #

# Graph the error and loss evolution.
import matplotlib.pyplot as plt
plt.plot(__error__)
plt.show()
# ====================================================================================================================================== #

