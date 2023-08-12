#All packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
data = pd.read_csv("heart.csv")
data.head()
data.info()
pp.ProfileReport(data)
#Model preparation
y = data["target"]
X = data.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#LightGBM Classifier
lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train, y_train)
lgbm_predicted = lgbm.predict(X_test)
lgbm_conf_matrix = confusion_matrix(y_test, lgbm_predicted)
lgbm_acc_score = accuracy_score(y_test, lgbm_predicted)

print("confusion matrix")
print(lgbm_conf_matrix)
print("\n")
print("Accuracy of LightGBM Classifier:", lgbm_acc_score * 100, '\n')
print(classification_report(y_test, lgbm_predicted))
# Logistic Regression
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)

print("confusion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:", lr_acc_score * 100, '\n')
print(classification_report(y_test, lr_predict))
# Extreme Gradient Boost
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15, gamma=0.6, subsample=0.52, colsample_bytree=0.6,
                    seed=27, reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)

print("confusion matrix")
print(xgb_conf_matrix)
print("\n")
print("Accuracy of Extreme Gradient Boost:", xgb_acc_score * 100, '\n')
print(classification_report(y_test, xgb_predicted))

#GNN Classifier
# Create graph structure (assuming a fully connected graph)
num_nodes = X_train.shape[0]
edge_index = torch.tensor([
    [i for i in range(num_nodes) for j in range(num_nodes)],
    [j for j in range(num_nodes) for i in range(num_nodes)]
], dtype=torch.long)

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Train GNN
gnn = GNNClassifier(num_features=X_train_tensor.shape[1], hidden_channels=64, num_classes=2)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
gnn.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = gnn(X_train_tensor, edge_index)
    loss = F.nll_loss(out, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions
gnn.eval()
with torch.no_grad():
    gnn_predicted = gnn(X_test_tensor, edge_index).max(1)[1]
gnn_conf_matrix = confusion_matrix(y_test, gnn_predicted)
gnn_acc_score = accuracy_score(y_test, gnn_predicted)

print("Confusion Matrix:")
print(gnn_conf_matrix)
print("\n")
print("Accuracy of GNN Classifier:", gnn_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, gnn_predicted))
# StackingCVClassifier
scv = StackingCVClassifier(classifiers=[xgb], meta_classifier=xgb, random_state=42)
scv.fit(X_train, y_train)
scv_predicted = scv.predict(X_test)
scv_conf_matrix = confusion_matrix(y_test, scv_predicted)
scv_acc_score = accuracy_score(y_test, scv_predicted)

print("confusion matrix")
print(scv_conf_matrix)
print("\n")
print("Accuracy of StackingCVClassifier:", scv_acc_score * 100, '\n')
print(classification_report(y_test, scv_predicted))

#Naive Bayes
m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpredicted = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpredicted)
nb_acc_score = accuracy_score(y_test, nbpredicted)
print("confussion matrix")
print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpredicted))

#Random Forest Classifier
m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

#K-nearest Neighbours
m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))

#Desicion tree 
m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))

#Support Vector Machine
m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))

lr_false_positive_rate, lr_true_positive_rate, lr_threshold = roc_curve(y_test, lr_predict)
xgb_false_positive_rate, xgb_true_positive_rate, xgb_threshold = roc_curve(y_test, xgb_predicted)
lgbm_false_positive_rate, lgbm_true_positive_rate, lgbm_threshold = roc_curve(y_test, lgbm_predicted)
gnn_false_positive_rate, gnn_true_positive_rate, gnn_threshold = roc_curve(y_test, gnn_predicted)
scv_false_positive_rate, scv_true_positive_rate, svc_threshold = roc_curve(y_test, scv_predicted)
rf_false_positive_rate, rf_true_positive_rate, rf_threshold = roc_curve(y_test, rf_predicted)
knn_false_positive_rate, knn_true_positive_rate, knn_threshold = roc_curve(y_test, knn_predicted)
dt_false_positive_rate, dt_true_positive_rate, dt_threshold = roc_curve(y_test, dt_predicted)
nb_false_positive_rate, nb_true_positive_rate, nb_threshold = roc_curve(y_test, nbpredicted)
svc_false_positive_rate, svc_true_positive_rate, svc_threshold = roc_curve(y_test, svc_predicted)

sns.set_style('whitegrid')
plt.figure(figsize=(10, 5))
plt.title('Receiver Operating Characteristic Curve')

plt.plot(lr_false_positive_rate, lr_true_positive_rate, label='Logistic Regression')
plt.plot(xgb_false_positive_rate, xgb_true_positive_rate, label='Extreme Gradient Boost')
plt.plot(lgbm_false_positive_rate, lgbm_true_positive_rate, label='LightGBM')
plt.plot(gnn_false_positive_rate, gnn_true_positive_rate, label='Graphical Neural Networks')
plt.plot(scv_false_positive_rate, scv_true_positive_rate, label='StackingCVClassifier')
plt.plot(rf_false_positive_rate, rf_true_positive_rate, label='Random Forest Classifier')
plt.plot(knn_false_positive_rate, knn_true_positive_rate, label='K-Nearest Neighbors')
plt.plot(dt_false_positive_rate, dt_true_positive_rate, label='Decision Tree Classifier')
plt.plot(nb_false_positive_rate, nb_true_positive_rate, label='Naive Bayes')
plt.plot(svc_false_positive_rate, svc_true_positive_rate, label='Support Vector Machine')

plt.plot([0, 1], ls='--', color='.5')
plt.plot([0, 0], [1, 0], c='.5')
plt.plot([1, 1], c='.5')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

# Model Evaluation
model_ev = pd.DataFrame({'Model': ['Logistic Regression', 'Extreme Gradient Boost', 'LightGBM',
                                    'Graphical Neural Networks', 'StackingCVClassifier', 'Random Forest', 
                                    'K-Nearest Neighbours', 'Descision Tree Classifier', 'Naive Bayes',
                                    'Support Vector Machine'],
                         'Accuracy': [lr_acc_score * 100, xgb_acc_score * 100, lgbm_acc_score * 100,
                                      gnn_acc_score * 100, scv_acc_score * 100, rf_acc_score * 100,
                                      knn_acc_score * 100, dt_acc_score * 100, nb_acc_score * 100,
                                      svc_acc_score * 100]})
colors = ['red', 'green', 'blue', 'gold', 'silver', 'purple', 'black', 'cyan', 'magenta', 'orange']
plt.figure(figsize=(12, 5))
plt.title("Barplot Representing Accuracy of Different Models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_ev['Model'], model_ev['Accuracy'], color=colors)
plt.xticks(rotation=45)
plt.show()
