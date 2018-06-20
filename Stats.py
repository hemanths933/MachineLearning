import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon,binom,poisson
#############################################
#Calculating mean,std-dev and variance
#############################################

numbers = np.random.normal(400.4,6.56,4)
variance=0
mn = np.mean(numbers)
print(numbers)
mean=0
for i in numbers:
    variance = variance + ((i*i)-(mn*mn))
    mean = mean + i
variance= variance/len(numbers)
mean = mean/len(numbers)
print("variance is ",variance)
std_dev = np.sqrt(variance)
print("std)dev is ",std_dev)
print("The mean is ",mean)
#############################################################
print("the numpy mean is ",np.mean(numbers))
print("the numpy variance is ",np.var(numbers))
print("the numpy std dev is ",np.std(numbers))

###############################################################
#-Visualizing the data
###############################################################
#probability density functions

#Normal distribution
numbers1 = np.random.normal(100,1,10000)#mean,std-dev,size_of_data
plt.hist(numbers1,50)#array,buckets
plt.show()

#uniform distribution
numbers1 = np.random.uniform(-10,10,10000)#startrange,endrange,size
numbers2 = np.arange(-10,10,0.001)#produces non random incremental of 0.001 values from 10 to -10
plt.hist(numbers1,50)#array,buckets
plt.show()

#Exponential distribution
numbers1 = np.random.exponential(1,10)
plt.hist(numbers1,3)
plt.show()

#to show how exponential dataset looks
numbers1 = np.arange(0,10,0.001)#start,end,increment
plt.plot(numbers1,expon.pdf(numbers1))#x-axis,y-axis
#plt.plot(numbers1,np.exp(numbers1))
plt.show()

###################################
#Probability Mass Functions
###################################

#binomial distribution
numbers1 = np.arange(0,10,0.5)
print("binomial data is ",binom.pmf(numbers1,10,0.5))#array,n,p
print("the length of binomial dataset",len(binom.pmf(numbers1,10,0.5)))#array,n,p
plt.plot(numbers1,binom.pmf(numbers1,10,0.5))#array,n,p
plt.show()

#poisson distribution
numbers1 = np.arange(0,10,0.5)
plt.plot(numbers1,poisson.pmf(numbers1,5))#array,mean
plt.show()