import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import cPickle as pickle
except:
    import pickle

with open("batch_norm_accuracy.p","rb") as f:
    batch_norm_accuracy = pickle.load(f)
    
with open("rbm_pretraining_accuracy.p","rb") as f:
    rbm_pretraining_accuracy = pickle.load(f)

with open("grad_desc_2_accuracy.p","rb") as f:
    grad_desc_2_accuracy = pickle.load(f)
    
with open("grad_desc_4_accuracy.p","rb") as f:
    grad_desc_4_accuracy = pickle.load(f)
    
with open("res_grad_desc_accuracy.p","rb") as f:
    res_grad_desc_accuracy = pickle.load(f)

with open("res_batch_norm_accuracy.p","rb") as f:
    res_batch_norm_accuracy = pickle.load(f)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(range(len(batch_norm_accuracy)),batch_norm_accuracy,alpha=0.5,s=2,c='b',edgecolor='')
ax1.scatter(range(len(rbm_pretraining_accuracy)),rbm_pretraining_accuracy,alpha=0.5,s=2,c='r',edgecolor='')
ax1.scatter(range(len(grad_desc_2_accuracy)),grad_desc_2_accuracy,alpha=0.5,s=2,c='c',edgecolor='')
ax1.scatter(range(len(grad_desc_4_accuracy)),grad_desc_4_accuracy,alpha=0.5,s=2,c='g',edgecolor='')
ax1.scatter(range(len(res_grad_desc_accuracy)),res_grad_desc_accuracy,alpha=0.5,s=2,c='m',edgecolor='')
ax1.scatter(range(len(res_batch_norm_accuracy)),res_batch_norm_accuracy,alpha=0.5,s=2,c='darkorange',edgecolor='')

s1 = mpatches.Patch(color='b', label='batch normalization')
s2 = mpatches.Patch(color='r', label='rbm pre-training')
s3 = mpatches.Patch(color='c', label='2-layer gradient descent')
s4 = mpatches.Patch(color='g', label='4-layer gradient descent')
s5 = mpatches.Patch(color='m', label='residual gradient descent')
s6 = mpatches.Patch(color='darkorange', label='residual batch norm')
plt.legend(handles=[s3,s4,s1,s5,s6,s2],loc='upper right')

plt.title("Test Set Accuracy")
plt.xlabel('Iteration')
plt.ylabel('Test Set Error')
plt.xlim(-100, 3000)
plt.savefig('test_accuracy_1.png')
plt.show()

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(range(len(batch_norm_accuracy))[200:],batch_norm_accuracy[200:],alpha=0.5,s=2,c='b',edgecolor='')
ax1.scatter(range(len(rbm_pretraining_accuracy))[200:],rbm_pretraining_accuracy[200:],alpha=0.5,s=2,c='r',edgecolor='')
ax1.scatter(range(len(grad_desc_2_accuracy))[200:],grad_desc_2_accuracy[200:],alpha=0.5,s=2,c='c',edgecolor='')
ax1.scatter(range(len(grad_desc_4_accuracy))[200:],grad_desc_4_accuracy[200:],alpha=0.5,s=2,c='g',edgecolor='')
ax1.scatter(range(len(res_grad_desc_accuracy))[200:],res_grad_desc_accuracy[200:],alpha=0.5,s=2,c='m',edgecolor='')
ax1.scatter(range(len(res_batch_norm_accuracy))[200:],res_batch_norm_accuracy[200:],alpha=0.5,s=2,c='darkorange',edgecolor='')

s1 = mpatches.Patch(color='b', label='batch normalization')
s2 = mpatches.Patch(color='r', label='rbm pre-training')
s3 = mpatches.Patch(color='c', label='2-layer gradient descent')
s4 = mpatches.Patch(color='g', label='4-layer gradient descent')
s5 = mpatches.Patch(color='m', label='residual gradient descent')
s6 = mpatches.Patch(color='darkorange', label='residual batch norm')
plt.legend(handles=[s3,s4,s1,s5,s6,s2],loc='upper right')

plt.title("Test Set Accuracy (Zoomed)")
plt.xlabel('Iteration')
plt.ylabel('Test Set Error')
plt.xlim(200, 3000)
plt.savefig('test_accuracy_2.png')
plt.show()