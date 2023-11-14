# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV

# Load your dataset (you need labeled data for training)
# Assume you have a CSV file with 'text' and 'label' columns
# 'text' column contains the news content, and 'label' column contains the binary label (0 for real news, 1 for fake news)
df = pd.read_csv('news.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#SVM accuracy
svm_classifier = SVC()
svm_classifier.fit(X_train_tfidf, y_train)
svm_pred = svm_classifier.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

#hyperparamater tunings
# Example with GridSearchCV for Naive Bayes
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

best_classifier = grid_search.best_estimator_
best_pred = best_classifier.predict(X_test_tfidf)

print("Best Model Accuracy:", accuracy_score(y_test, best_pred))
print("\nBest Model Classification Report:\n", classification_report(y_test, best_pred))
print("\nBest Model Confusion Matrix:\n", confusion_matrix(y_test, best_pred))