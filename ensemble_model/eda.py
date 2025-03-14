import pandas as pd

# Load CSV files into DataFrames
train_essays = pd.read_csv('./data/llm-detect-ai-generated-text/train_essays.csv')
ai_ga_dataset = pd.read_csv('./data/ai-ga-dataset.csv')
train_v2_drcat_02 = pd.read_csv('./data/train_v2_drcat_02.csv')
training_essay_data = pd.read_csv('./data/Training_Essay_Data.csv')

def summarize_df(df, df_name):
    """
    Prints the list of columns and the total number of rows in the given DataFrame.
    """
    print(f"===== {df_name} =====")
    print(f"Column list: {df.columns.tolist()}")
    print(f"Number of rows: {len(df)}")
    print()

# Print summary information for each DataFrame
summarize_df(train_essays, 'train_essays.csv')
summarize_df(ai_ga_dataset, 'ai-ga-dataset.csv')
summarize_df(train_v2_drcat_02, 'train_v2_drcat_02.csv')
summarize_df(training_essay_data, 'Training_Essay_Data.csv')