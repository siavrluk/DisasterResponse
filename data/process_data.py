import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='left', on='id')
       
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns and fix their names
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.str.split('-').str[0]
    categories.columns = category_colnames
    
    # convert category values to just numbers
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    
    # make sure the mew category columns contain just 0 and 1
    for column in categories:
        df.drop(df[~df[column].isin([0,1 ])].index, inplace=True)
        
    return df
    
    
def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse_clean.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()