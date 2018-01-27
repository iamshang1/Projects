#import numpy
import numpy as np

#create numpy array
x = [[1,2,3],[4,5,6]]
x = np.array(x)
print(x)

#array info
print(x.shape)
print(x.dtype)

#indexing
print(x[0,0])
print(x[:,0])
print(x[0,:])
print(x[0,1:3])

#setting values
print(x)
x[1,2] = 100
print(x)
x[1,:] = [101,102,103]
print(x)

#summary statistics
x = [[1,2,3],[4,5,6]]
x = np.array(x)
print(x.mean())
print(x.std())
print(x.min())
print(x.max())

#statistics for columns and rows
print(np.mean(x,0))
print(np.mean(x,1))

#transformations
x_transpose = x.T
print(x_transpose)

x_reshape = x.reshape(3,2)
print(x_reshape)

#other ways to intialize arrays
ones = np.ones(shape=(3,5))
print(ones)

zeros = np.zeros(shape=(3,5))
print(zeros)

random =  np.random.rand(3,5)
print(random)

rand_ints = np.random.randint(low=0,high=10,size=(3,5))
print(rand_ints)

seq = np.arange(start=0,stop=3,step=0.5)
print(seq)

#operations
a = np.arange(0,10).reshape(2,5)
b = np.arange(5,15).reshape(2,5) 
elem_sum = a + b
print(elem_sum)

print(a + 1)
print(a + np.random.rand(2,1))
print(a + np.random.rand(1,5))

elem_product = a * b
print(elem_product)

print(a * 2)

dot = np.dot(a, b.T)
print(dot)

#normalize
x = np.random.rand(3,5)
x -= np.mean(x,0)
x /= np.std(x,0)
print(x)
print(np.mean(x,0),np.std(x,0))

#concatenating arrays
a = np.zeros((2,5))
b = np.ones((2,5))
c = np.ones((2,5)) * 2
print(np.vstack((a,b,c)))
print(np.vstack((c,b,a)))
print(np.hstack((a,b,c)))
print(np.hstack((c,b,a)))
print(np.concatenate((a,b,c),axis=0))
print(np.concatenate((a,b,c),axis=1))

#filtering and masking
a = np.random.rand(20)
print(a)
filter = a > 0.5
print(filter)
print(a[filter])