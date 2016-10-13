import numpy as np
import theano
import theano.tensor as T

#rnn parameters
learning_rate = 1
input = T.matrix()
W1 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(2,10)),dtype=theano.config.floatX))
Wh = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(10,10)),dtype=theano.config.floatX))
W2 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(10,1)),dtype=theano.config.floatX))
b1 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(10)),dtype=theano.config.floatX))
b2 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(1)),dtype=theano.config.floatX))
h0 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(10)),dtype=theano.config.floatX))
target = T.vector()

def step(input,h0,W1,Wh,W2,b1,b2):
    h_out = T.nnet.sigmoid(T.dot(input,W1)+T.dot(h0,Wh)+b1)
    return h_out

params = [W1,Wh,W2,b1,b2,h0]
h_output,_ = theano.scan(fn=step,sequences=input,outputs_info=h0,non_sequences=params[:-1])
output =  T.nnet.sigmoid(T.dot(h_output,W2)+b2)
cost = T.nnet.binary_crossentropy(output.flatten(1),target).mean()
grads = [T.grad(cost, param) for param in params]
train = theano.function([input,target],cost,updates=[(param,param-learning_rate*grad) for param,grad in zip(params,grads)])
predict = theano.function([input],output)

#train
for i in range(25000):
    i1 = np.random.randint(100000000,1000000000)
    i2 = np.random.randint(100000000,1000000000)
    o1 = i1+i2
    i1 = map(int,bin(i1)[2:])
    i2 = map(int,bin(i2)[2:])
    o1 = map(int,bin(o1)[2:])
    if len(i1)>len(i2):
        diff = len(i1)-len(i2)
        i2 = [0]*diff+i2
    elif len(i2)>len(i1):
        diff = len(i2)-len(i1)
        i1 = [0]*diff+i1
    if len(o1)>len(i1):
        i1 = [0]+i1
        i2 = [0]+i2
    i1.reverse()
    i2.reverse()
    o1.reverse()
    X = np.vstack((i1,i2)).T
    y = np.array(o1)
    print "step %i training error:" % (i+1), train(X,y)

#test
for i in range(100):
    in1 = np.random.randint(100000000,1000000000)
    in2 = np.random.randint(100000000,1000000000)
    actual = in1+in2
    i1 = map(int,bin(in1)[2:])
    i2 = map(int,bin(in2)[2:])
    o1 = map(int,bin(actual)[2:])
    if len(i1)>len(i2):
        diff = len(i1)-len(i2)
        i2 = [0]*diff+i2
    elif len(i2)>len(i1):
        diff = len(i2)-len(i1)
        i1 = [0]*diff+i1
    if len(o1)>len(i1):
        i1 = [0]+i1
        i2 = [0]+i2
    i1.reverse()
    i2.reverse()
    X = np.vstack((i1,i2)).T
    pred = predict(X)
    pred = int(''.join(list(reversed([str(int(round(p))) for p in pred]))),2)
    print "%i + %i: pred: %i actual: %i " % (in1, in2, pred, actual)