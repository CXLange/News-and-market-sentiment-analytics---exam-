import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


# Text standardizing for title and text columns

data['title'] = data['title'].astype(str).str.strip().str.lower()

data['text'] = data['text'].astype(str).str.strip().str.lower()