##Source Code
 
##Applying SVM Algorithm
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix 

# Load the dataset 
df = pd.read_csv('data_csv.csv') 

# Remove duplicate rows 
df.drop_duplicates(inplace=True) 

# Handle missing values  
for col in df.columns: 
if df[col].dtype == 'object': 
df[col].fillna(df[col].mode()[0], inplace=True) 
else: 
df[col].fillna(df[col].median(), inplace=True) 

# Encode categorical variables 
label_encoders = {} 
for col in df.select_dtypes(include=['object']).columns: 
le = LabelEncoder() 
df[col] = le.fit_transform(df[col]) 
label_encoders[col] = le 
print(f'Label Encoder mapping for {col}: {dict(zip(le.classes_, 
le.transform(le.classes_)))}') 

# Define features and target variable 
X = df.drop(columns=['class']) 
y = df['class'] 

# Split dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42)  

# Normalize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

# Train SVM model 
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) 
svm_model.fit(X_train, y_train) 

# Make predictions 
y_pred = svm_model.predict(X_test) 

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
print(classification_report(y_test, y_pred)) 

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", conf_matrix) 

# Plot confusion matrix 
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=label_encoders['class'].classes_, 
yticklabels=label_encoders['class'].classes_) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show() 

# data points - Scatter plot with hue 
plt.figure(figsize=(8,6)) 
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='coolwarm', 
edgecolor='k', alpha=0.7) 
plt.xlabel('Learning disorder') 
plt.ylabel('Genetic_disorders') 
plt.title('Scatter Plot of Learning disorder vs Genetic_disorders') 
plt.show()
 
##Applying Decision tree 

#Normalize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
 

# Train Decision Tree model 
dt_model = DecisionTreeClassifier(random_state=42) 
dt_model.fit(X_train, y_train) 

# Make predictions 
y_pred = dt_model.predict(X_test) 

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
print(classification_report(y_test, y_pred)) 

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", conf_matrix) 

# Plot confusion matrix 
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=label_encoders['class'].classes_, 
yticklabels=label_encoders['class'].classes_) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show() 

##Applying Random Forest 

# Check class balance 
print("\nClass distribution:") 
print(df['class'].value_counts()) 

# Optional: Add slight noise to numeric features to make the model generalize 
better 
for col in X.select_dtypes(include=[np.number]).columns: 
X[col] += np.random.normal(0, 0.01, size=X.shape[0] 

# Normalize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

# Train Random Forest  
rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, 
random_state=42) 
rf_model.fit(X_train, y_train)  

# Make predictions 
y_pred = rf_model.predict(X_test) 

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
print(classification_report(y_test, y_pred)) 

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", conf_matrix) 

# Plot confusion matrix 
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=label_encoders['class'].classes_, 
yticklabels=label_encoders['class'].classes_) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show() 

##Applying Naïve Bayes 

# Normalize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

# Train Naive Bayes model 
nb_model = GaussianNB() 
nb_model.fit(X_train, y_train) 

# Make predictions 
y_pred = rf_model.predict(X_test) 

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
print(classification_report(y_test, y_pred)) 

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix) 

# Plot confusion matrix 
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=label_encoders['class'].classes_, 
yticklabels=label_encoders['class'].classes_) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show()
 
##Applying Logistic Regression 

# Normalize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

# Train Logistic Regression model 
lr_model = LogisticRegression(random_state=42, max_iter=1000) 
lr_model.fit(X_train, y_train) 

# Make predictions 
y_pred = lr_model.predict(X_test) 

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
print(classification_report(y_test, y_pred)) 

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", conf_matrix) 

# Plot confusion matrix 
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=label_encoders['class'].classes_, 
yticklabels=label_encoders['class'].classes_) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix') 
plt.show()