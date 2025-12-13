import numpy as np
import pandas as pd

true_data = pd.read_csv('data/True.csv')
fake_data = pd.read_csv('data/Fake.csv')


# Creating a single dataset with labels for true = 1 and false = 0

true_data['label'] = 1
fake_data['label'] = 0

# Cleaning true_data before concatinating
## Removing journal identifier
true_data['text'] = true_data['text'].str.partition('- ')[2]

# Removing duplicates rows of titles where text is empty
# Create a 
is_text_functionally_empty = (
    true_data['text'].isna() | 
    true_data['text'].astype(str).str.strip().eq('')
)
rows_to_drop = true_data[is_text_functionally_empty].duplicated(subset=['title'], keep='first')
drop_indices = true_data[is_text_functionally_empty][rows_to_drop].index
cleaned_true = true_data.drop(index=drop_indices)
# Doing the same for fake articles
fake_is_text_functionally_empty = (
    fake_data['text'].isna() | 
    fake_data['text'].astype(str).str.strip().eq('')
)
rows_to_drop = fake_data[fake_is_text_functionally_empty].duplicated(subset=['title'], keep='first')
drop_indices = fake_data[fake_is_text_functionally_empty][rows_to_drop].index
cleaned_fake = fake_data.drop(index=drop_indices)

print(f"Removed {len(true_data)-len(cleaned_true)} duplicate title rows from true articles")
print(f"Removed {len(fake_data)-len(cleaned_fake)} duplicate title rows from fake articles")
print("")

# Removing duplicates of text but keeping unique rows of title
has_content = ~(
    cleaned_fake['text'].isna() |
    cleaned_fake['text'].astype(str).str.strip().eq('')
)
rows_to_remove = cleaned_fake[has_content].duplicated(subset=['text'], keep='first')

drop_indices = cleaned_fake[has_content][rows_to_remove].index

new_cleaned_fake = cleaned_fake.drop(index=drop_indices)

# Doing the same for true articles

has_content = ~(
    cleaned_true['text'].isna() |
    cleaned_true['text'].astype(str).str.strip().eq('')
)
rows_to_remove = cleaned_true[has_content].duplicated(subset=['text'], keep='first')

drop_indices = cleaned_true[has_content][rows_to_remove].index

new_cleaned_true = cleaned_true.drop(index=drop_indices)

print(f"Removed {len(cleaned_true)-len(new_cleaned_true)} duplicate text rows from true articles")
print(f"Removed {len(cleaned_fake)-len(new_cleaned_fake)} duplicate text rows from fake articles")
print("______________________________________________________________________")
print(f"Removed {len(true_data)-len(new_cleaned_true)} duplicate rows from true articles in total")
print(f"Removed {len(fake_data)-len(new_cleaned_fake)} duplicate rows from fake articles in total")

# Concatinating dataframes
data = pd.concat([cleaned_true, cleaned_fake])


# Text standardizing
import string


# Creating new columns to preserve original text
data['title_standard'] = data['title']
data['text_standard'] = data['text']

# Removing punctuations including special letter which didn't get picked up by string.punctuation
punctuation_and_special = string.punctuation + '“”‘’' 
punctuation = str.maketrans('', '', punctuation_and_special)

data['title_standard'] = data['title'].astype(str).str.translate(punctuation)
data['text_standard'] = data['text'].astype(str).str.translate(punctuation)

# Lowercasing 
data['title_standard'] = data['title_standard'].astype(str).str.strip().str.lower()
data['text_standard'] = data['text_standard'].astype(str).str.strip().str.lower()


# Baseline naive bayes model for article texts
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Define Features (X) and Target (y)
X = data['text_standard']  
y = data['label']

# Splitting the data and making sure we get the same number of labels in each set with stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y
)

# Setting max_df to 0.7 we exclude words that appears in more than 70% of all articles in the training set so we should get more unique words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
X_train_counts = count_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_counts = count_vectorizer.transform(X_test)

# Train Multinomial Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# 5. Predict and Evaluate
y_pred_counts = nb_classifier.predict(X_test_counts)

# Get the feature names to see which words are helping us predict
feature_names = count_vectorizer.get_feature_names_out()

# Getting log coefficients 
log_probs_fake = nb_classifier.feature_log_prob_[0]
log_probs_true = nb_classifier.feature_log_prob_[1]

# Create a DataFrame so we can sort them 
feature_df = pd.DataFrame({
    'feature': feature_names,
    'log_prob_fake': log_probs_fake,
    'log_prob_true': log_probs_true
})

# Calculate the difference between coefficients 
# A larger positive difference means the word is highly associated with fake news
feature_df['fake_score'] = feature_df['log_prob_fake'] - feature_df['log_prob_true']

# 5. Get top 20 for Fake (highest positive scores)
top_fake_features = feature_df.sort_values(by='fake_score', ascending=False).head(20)

# 6. Get top 20 for True (lowest negative scores)
top_true_features = feature_df.sort_values(by='fake_score', ascending=True).head(20)


# Print results
print("--- Naive Bayes Classification Results (Using Raw Word Counts) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_counts):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_counts))

# Print important classifying words
print("--- Top 20 Words Predicting FAKE News (Label 0) ---")
print(top_fake_features[['feature', 'fake_score']].to_markdown(index=False))

print("\n--- Top 20 Words Predicting TRUE News (Label 1) ---")
print(top_true_features[['feature', 'fake_score']].to_markdown(index=False))


# We do the same for the article titles

X = data['title_standard']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
count_vectorizer = CountVectorizer(stop_words='english')


X_train_counts = count_vectorizer.fit_transform(X_train)


X_test_counts = count_vectorizer.transform(X_test)


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)


y_pred_counts = nb_classifier.predict(X_test_counts)

feature_names = count_vectorizer.get_feature_names_out()

log_probs_fake = nb_classifier.feature_log_prob_[0]
log_probs_true = nb_classifier.feature_log_prob_[1]


feature_df = pd.DataFrame({
    'feature': feature_names,
    'log_prob_fake': log_probs_fake,
    'log_prob_true': log_probs_true
})

feature_df['fake_score'] = feature_df['log_prob_fake'] - feature_df['log_prob_true']
top_fake_features = feature_df.sort_values(by='fake_score', ascending=False).head(20)
top_true_features = feature_df.sort_values(by='fake_score', ascending=True).head(20)

print("--- Naive Bayes Classification Results (Using Raw Word Counts) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_counts):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_counts))
print("--- Top 20 Words Predicting FAKE News (Label 0) ---")
print(top_fake_features[['feature', 'fake_score']].to_markdown(index=False))
print("\n--- Top 20 Words Predicting TRUE News (Label 1) ---")
print(top_true_features[['feature', 'fake_score']].to_markdown(index=False))


# Added cleaning step to handle noise observed in fake articles
data['text_cleaned'] = data['text_standard']
# Find all words that consist only of letters (a-z) and more than 2 characters long to get rid of fx 21Wire
data['text_standard'] = data['text_cleaned'].str.findall(r'[a-z]{2,}')
# Join the tokenized words into single string again
data['text_standard'] = data['text_standard'].str.join(' ')