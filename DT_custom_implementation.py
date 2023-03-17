# %%
import numpy as np
import pandas as pd

# %%
data = pd.read_csv("../data_used/DT_titanic.csv")
data.head()

# %%
data.info()

# %%
colums_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked']
new_data = data.drop(colums_to_drop , axis = 1)

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
new_data['Sex'] = le.fit_transform(new_data['Sex'])

# %%
clean_data = new_data.fillna(new_data['Sex'].mean()) # for making sex colunm entry equals to other column entry
clean_data.info()

# %%
from sklearn.model_selection import train_test_split
input_column = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
output_column = ['Survived']
X = clean_data[input_column]
y = clean_data[output_column]
X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.33,random_state=42)


# %%
def entropy(col):
    data,counts = np.unique(col , return_counts=True)
    entr = 0 
    for count in counts:
        probab = float(count) /col.shape[0]
        entr += probab*np.log2(probab)
    return -entr

# %%
def divide_data(X_data , feature_key , featur_value):
    X_left = pd.DataFrame([] , columns=X_data.columns)
    X_right = pd.DataFrame([] , columns=X_data.columns)
    for xi in range(X_data.shape[0]):
        val = X_data[feature_key].iloc[xi] # here we are locating the value of each row alonf feature key
        if val<featur_value:
            X_left.append(X_data.loc[xi]) # constructing data_frame in right of tree
        else:
            X_right.append(X_data.loc[xi])# constructing data_frame in right of tree
    return X_left, X_right

# %%
def information_gain(X_data,f,val):
    left,right = divide_data(X_data,f,val)
    hs = entropy(X_data.Survived)
    left_pro = (left.shape[0])/float(X_data.shape[0])
    right_pro = (right.shape[0])/float(X_data.shape[0])
    igain = hs - (left_pro*entropy(left.Survived)+right_pro*entropy(right.Survived))
    return igain

# %%
for f in (X.columns):
    print(f)
    # print(information_gain(clean_data,f,clean_data[f].mean()))

# %% [markdown]
# CUSTOM DT

# %%
class DT:
    def __init__(self , depth = 0 , max_depth = 5):
        self.depth = depth
        self.max_depth = max_depth
        self.target = None
        self.fkey = None
        self.fvalue = None
        self.left = None
        self.right = None
    def fit(self,X_train):
        i_gain=[]
        # features = ['Pclass','Sex','Age','Fare']
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        for f in features:
            ig = information_gain(X_train, f, X_train[f].mean())
            i_gain.append(ig)
        self.fkey = features[np.argmax(i_gain)]
        self.fvalue = X_train[self.fkey].mean()
        left_tree,right_tree = divide_data(X_train,self.fkey,self.fvalue)

        left_tree = left_tree.reset_index(drop=True)
        right_tree = right_tree.reset_index(drop=True)

        # base condition 
        if left_tree.shape[0] == 0 or right_tree.shape[0] == 0:
            if X_train.Survived.mean()>=0.5:
                self.target = "survived"
            else :
                self.target = "dead"
            return
        if self.depth >= self.max_depth:
            if X_train.Survived.mean()>=0.5:
                self.target = "survived"
            else :
                self.target = "dead"
            return
        
        # recursive call
        self.left = DT(depth=self.depth+1)
        self.left.fit(left_tree)

        self.right = DT(depth=self.depth+1)
        self.right.fit(right_tree)

        # setting target to every node not just in leaf nodes because there can be a case where we cant get upto leaf nodes

        if X_train.Survived.mean()>=0.5:
            self.target = "survived"
        else :
            self.target = "dead"
        return
    
    def predict(self,X1):
        if X1[self.fkey]>self.fvalue :
            # go to right 
            if self.right is None :
                return self.target
            else:
                return self.right.predict(X1)
        else:
            if self.left is None :
                return self.target
            else:
                return self.left.predict(X1)

# %%
dt = DT()
# dt.fit(X_train)
# print(dt.fkey,dt.fvalue)split = int(0.7*clean_data.shape[0])
split = int(0.7*clean_data.shape[0])
train_data = clean_data[:split]
test_data = clean_data[split:]
test_data = test_data.reset_index(drop=True)

# %%
dt.fit(train_data)
y_pred = []
for i in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[i]))
y_pred[:10]

# %%
data[split:][:10]

# %%



