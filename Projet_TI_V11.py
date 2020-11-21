# Imports
import numpy as np 
import pickle as pk
import time
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow, show, colorbar
from sklearn import tree
from sklearn.metrics import confusion_matrix
#%% Open the source file
train_set, validation_set, test_set = pk.load(open('/Users/zarmi/Desktop/Uni-ISH/TI/mnist.pkl', 'rb'), encoding='latin1')
#%% Defining train and test arrays
train_samples=train_set[0]
test_samples=test_set[0]
train_labels=train_set[1]
test_labels=test_set[1]
#%% Reashape to unflatten
train_labels=np.reshape(train_labels, (len(train_labels),1))
test_labels = np.reshape(test_labels,(len(test_labels),1))
#%% Number of the data and classes
num_train_samples=np.shape(train_samples)[0]
num_test_samples=np.shape(test_samples)[0]
num_features=np.shape(train_samples)[1]
num_classes=10
#%% Train the decision tree
t0=[]
t1=[]
t2=[]
acc_test=[]
acc_train=[]
TREE=[]
num_leaves=[]
num_nodes=[]

for depth in range(3,51):
    t0.append(time.perf_counter())
    print(depth)
    # depth=10
    dtree= tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    dtree.fit(train_samples, train_labels)
    TREE.append(dtree)
    t1.append( time.perf_counter())
    num_leaves.append(dtree.get_n_leaves())
    num_nodes.append(dtree.tree_.node_count)
    # Use the Trained Decision Tree to Predict the Labels for Training and Test data
    train_pred=dtree.predict(train_samples)
    test_pred=dtree.predict(test_samples)
    t2.append(time.perf_counter())
    # Calculate the Accuracy of the Trained Decision Tree
    acc_test.append(np.sum(test_pred==test_labels.flatten())/num_test_samples)
    acc_train.append(np.sum(train_pred==train_labels.flatten())/num_train_samples)
    print('Training accuracy is ' + str(acc_train[-1]*100) + ' percent')
    print('Test accuracy is ' + str(acc_test[-1]*100) + ' percent')

#%% plot the Accuracy of the Trained Decision Tree
plt.plot(np.array(range(3,51)),np.array(acc_test[0:5])*100)
plt.plot(np.array(range(3,51)),np.array(acc_train[0:5])*100)
plt.legend(["Test Data", "Train Data"])
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy [%]')
plt.grid(b=None, which='major', axis='both')
plt.show()
plt.clf()
#%%
plt.plot(np.array(range(3,51)),np.array(t1[0:5])-np.array(t0[0:5]))
plt.xlabel('Tree Depth')
plt.ylabel('Training Time [s]')
plt.grid(b=None, which='major', axis='both')
plt.show()
plt.clf()
#%%
plt.plot(np.array(range(3,51)),(np.array(t2[0:5])-np.array(t1[0:5]))*1000000/60000)
plt.xlabel('Tree Depth')
plt.ylabel('Test Time [us]')
plt.grid(b=None, which='major', axis='both')
plt.show()
plt.clf()
#%%
plt.plot(np.array(range(3,51)),np.array(num_leaves[0:5])/1000)
plt.plot(np.array(range(3,51)),np.array(num_nodes)[0:5]/1000)
plt.legend(["Number of Leaves", "Number of Nodes"])
plt.xlabel('Tree Depth')
plt.ylabel('number [k]')

plt.grid(b=None, which='major', axis='both')
plt.show()
plt.clf()
#%% Calculate the Confusion Matrix
# y axis is true labels and x axis is for predicted labels
cm=confusion_matrix(test_labels,test_pred)
fig, ax = plt.subplots(figsize=(5,5))
cmap=plt.cm.Blues
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
               ha="center", va="center",
               color="white" if cm[i, j] > thresh else "black")
plt.show()
#%% See the one that is predicted incorrect (The true label is 0 but the algorithms has predicted it as 1)
check01=np.logical_and(test_labels.flatten()==0,test_pred==1)
result = np.where(check01.flatten() == True)
plt.imshow(test_samples[result[0][0],:].reshape(28,28),cmap='gray')
#%% Show The first Digit of Data Base
imshow(train_samples[49999:50000,:].reshape(28,28),cmap='gray')
plt.axis('off')
plt.show()


dtree.get_params()

