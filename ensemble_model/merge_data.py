import pandas as pd
import nltk

# Download 'punkt' if not already done (necessary for sentence tokenization)
nltk.download('punkt')

# 1. Read CSV files
train_essays = pd.read_csv('./data/llm-detect-ai-generated-text/train_essays.csv')
ai_ga_dataset = pd.read_csv('./data/ai-ga-dataset.csv')
train_v2_drcat_02 = pd.read_csv('./data/train_v2_drcat_02.csv')
training_essay_data = pd.read_csv('./data/Training_Essay_Data.csv')

# 2. Keep only (text, label) columns
#    - From train_essays.csv, keep 'text' and rename 'generated' to 'label'
df_essays = train_essays[['text', 'generated']].rename(columns={'generated': 'label'})

#    - From ai-ga-dataset.csv, rename 'abstract' to 'text' and keep 'label'
df_ai_ga = ai_ga_dataset[['abstract', 'label']].rename(columns={'abstract': 'text'})

#    - From train_v2_drcat_02.csv, columns (text, label) already exist, so use them directly
df_v2_drcat_02 = train_v2_drcat_02[['text', 'label']]

#    - From Training_Essay_Data.csv, keep 'text' and rename 'generated' to 'label'
df_training_essay = training_essay_data[['text', 'generated']].rename(columns={'generated': 'label'})

# 3. Concatenate dataframes vertically (row-wise)
merged_df = pd.concat(
    [df_essays, df_ai_ga, df_v2_drcat_02, df_training_essay],
    ignore_index=True
)

# 4. (Optional) Save the intermediate merged dataframe if needed
# merged_df.to_csv('merged_text_label.csv', index=False)
# print("Saved merged_text_label.csv successfully.")

print(merged_df.head())
print("Total number of rows after merging:", len(merged_df))

# --------------------------------------------------
# Below is the code for sentence splitting using NLTK
# --------------------------------------------------

# 5. Prepare a list to store the sentence-split rows
sent_rows = []

# 6. Perform sentence tokenization and build a new dataframe
for idx, row in merged_df.iterrows():
    text = row['text']
    label = row['label']
    
    # Use nltk to split text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    
    # Create a new row for each sentence
    for sent in sentences:
        sent_rows.append({'text': sent.strip(), 'label': label})

# 7. Convert the list to a dataframe
sent_df = pd.DataFrame(sent_rows, columns=['text', 'label'])

print(sent_df.head())
print("Total number of rows after sentence splitting:", len(sent_df))

# 8. Save the final CSV
sent_df.to_csv('./data/merged_text_label_per_sentence.csv', index=False)
print("File saved as merged_text_label_per_sentence.csv.")