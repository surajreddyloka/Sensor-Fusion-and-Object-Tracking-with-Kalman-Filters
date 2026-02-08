import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sb
from scipy.stats import norm
import time

# Despite noisy measurement of individual sensors, We can calculate an optimal estimate of all conditions.
# Plot the Distributions in this range:

x = np.linspace(-100,100,1000)
mean0 = 0.0
var0 = 20.0
 
plt.figure(figsize = (10,5))
plt.plot(x, norm.pdf(x, mean0, var0), label='Normal Distribution')
plt.ylim(0, 0.1)
plt.legend(loc = 'best')
plt.xlabel('Position')

# Now we have something, which estimates the moved distance
# The Mean is meters, calculated from velocity*dt
# VarMove is the Estimated or determined with static measurements

meanMove = 25.0
varMove = 10.0

plt.figure(figsize = (10,5))
plt.plot(x, norm.pdf(x, meanMove, varMove), label='Normal Distribution')
plt.ylim(0, 0.1)
plt.legend(loc='best')
plt.xlabel('Distance moved')

# Both Distributions have to be merged together

def predict(var, mean, varMove, meanMove):
    new_var = var + varMove
    new_mean= mean+ meanMove
    return new_var, new_mean

new_var, new_mean = predict(var0, mean0, varMove, meanMove)

plt.figure(figsize=(10,5))
plt.plot(x,norm.pdf(x, mean0, var0), label='Beginning Normal Distribution')
plt.plot(x,norm.pdf(x, meanMove, varMove), label='Movement Normal Distribution')
plt.plot(x,norm.pdf(x, new_mean, new_var), label='Resulting Normal Distribution')
plt.ylim(0, 0.1)
plt.legend(loc='best')
plt.title('Normal Distributions of Kalman Filter Prediction Step')
plt.show()

# plt.savefig('Kalman-Filter-1D-Step.png', dpi=150)

# Sensor Defaults for Position Measurements
# Estimates or Determined with static measurements

meanSensor = 25.0
varSensor  = 12.0

plt.figure(figsize=(10,5))
plt.plot(x,norm.pdf(x, meanSensor, varSensor))
plt.ylim(0, 0.1)

# Now both Distributions have to be merged together

def correct(var, mean, varSensor, meanSensor):
    new_mean=(varSensor*mean + var*meanSensor) / (var+varSensor)
    new_var = 1/(1/var +1/varSensor)
    return new_var, new_mean

var, mean = correct(new_var, new_mean, varSensor, meanSensor)

plt.figure(figsize=(10,5))
plt.plot(x,norm.pdf(x, new_mean, new_var), label='Beginning (after Predict)')
plt.plot(x,norm.pdf(x, meanSensor, varSensor), label='Position Sensor Normal Distribution')
plt.plot(x,norm.pdf(x, mean, var), label='New Position Normal Distribution')
plt.ylim(0, 0.1)
plt.legend(loc='best')
plt.title('Normal Distributions of 1st Kalman Filter Update Step')


# Put everything in 1D Kalman Filter

positions = (10, 20, 30, 40, 50)+np.random.randn(5)
distances = (10, 10, 10, 10, 10)+np.random.randn(5)

for m in range(len(positions)):
    
    # Predict
    var, mean = predict(var, mean, varMove, distances[m])
    #print('mean: %.2f\tvar:%.2f' % (mean, var))
    plt.plot(x,norm.pdf(x, mean, var), label='%i. step (Prediction)' % (m+1))
    
    # Correct
    var, mean = correct(var, mean, varSensor, positions[m])
    print('After correction:  mean= %.2f\tvar= %.2f' % (mean, var))
    plt.plot(x,norm.pdf(x, mean, var), label='%i. step (Correction)' % (m+1))
    
plt.ylim(0, 0.1)
plt.xlim(-20, 120)
plt.legend()
plt.show()






