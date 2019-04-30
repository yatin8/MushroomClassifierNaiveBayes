import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df=pd.read_csv('./mushroomData/mushrooms.csv')
# print(df.head(10))
le=LabelEncoder()
ds=df.apply(le.fit_transform)
# print(ds.head(10))
data=ds.values;
# print(data[:10,:])
# print(type(data))
train,test=train_test_split(data,test_size=0.2)
# print(train.shape)
# print(test.shape)
x_train=train[:,1:]
y_train=train[:,0]

x_test=test[:,1:]
y_test=test[:,0]


def prior_prob(y_train,label):
    total_examples=y_train.shape[0]
    class_examples=np.sum(y_train==label)
    return float(class_examples)/float(total_examples)

# x=np.array([4,5,6,7,1,2,3])
# y=np.array([0,1,1,1,0,0,1])
# print(x[y==0])


def conditional_prob(x_train,y_train,feature_col,feature_value,label):
    x_filtered=x_train[y_train == label]
    numerator=np.sum(x_filtered[:,feature_col]==feature_value)
    denominator=np.sum(y_train == label)
    return float(numerator)/float(denominator)

def posterior_prob_max(x_train,y_train,x_test):
    classes=np.unique(y_train)
    n_features=x_train.shape[1]
    posterior_prob=[]
    for label in classes:
        likelihood=1.0
        for feature in range(n_features):
            conditional=conditional_prob(x_train,y_train,feature,x_test[feature],label)
            likelihood=likelihood*conditional

        prior=prior_prob(y_train,label)
        posterior=likelihood*prior
        posterior_prob.append(posterior)

    prediction=np.argmax(posterior_prob)
    return prediction

test=100
output=posterior_prob_max(x_train,y_train,x_test[test])
# print(output)
# print(y_test[test])

def score(x_train,y_train,x_test,y_test):
    pred=[]
    for i in range(x_test.shape[0]):
        prediction=posterior_prob_max(x_train, y_train, x_test[i])
        pred.append(prediction)

    pred=np.array(pred)
    accuracy=np.sum(pred==y_test)/y_test.shape[0]
    return accuracy

accuracy=score(x_train,y_train,x_test,y_test)
print(accuracy)