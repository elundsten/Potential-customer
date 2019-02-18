import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

data = pd.read_excel("Dustin_2.xlsx")
data = data.drop('OrganisationNo', axis=1)
#Removing all NaN rows
data = data.dropna(axis=0, how='any')
#Removing all duplicates
data = data.drop_duplicates()
target = data['Software']
train = data.drop('Software', axis=1)
#Converting all pandas "object" to numeric values in terms of categories

train['LineDiscount'] = train['LineDiscount'].astype('category').cat.codes
train['CustomerUnit'] = train['CustomerUnit'].astype('category').cat.codes
train['AccountManager'] = train['AccountManager'].astype('category').cat.codes
train['Municipality'] = train['Municipality'].astype('category').cat.codes
train['County'] = train['County'].astype('category').cat.codes
train['TargetAccessory'] = train['TargetAccessory'].astype('category').cat.codes
train['Country'] = train['Country'].astype('category').cat.codes
target = target.astype('category').cat.codes
seed = 102

#splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state = seed)

#Initialize and train the model
model = xgb.XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Evaluation Accuracy,precision, recall and confusion matrix
predictions = [round(value) for value in y_pred]
# print("First 20 predictions: ", predictions[0:100])
# print("First 20 real values: ", y_test[0:100])
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("precision: %.2f%%" % (precision * 100.0))
cm = confusion_matrix(y_test,predictions)
class_names = ["0","1"]
plot_confusion_matrix(cm,classes = class_names,title="confusion_matrix")
#Index(['CustomerUnit', 'Municipality', 'County'], dtype='object')