import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function reads the raw data from messages and categories csv 
    files into dataframes and merges the two separate dataframes as one.

    Args:
        messages_filepath: Path to the messages csv file
        categories_filepath: Path to the categories csv file

    Returns:
        A pandas dataframe containing messages and categories dataframe
        merged as a single dataframe joined together using 'id' column.

    Raises:
        The IOException: No such file or directory.
        If any of the filepaths are invalid or non-existent.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    This function performs data cleaning and transformation.
    It should be called after 'load_data' method is called.
    Args:
        df: A pandas dataframe with raw data.
    
    Returns:
        df: A clean pandas dataframe after applying the specific
        transformations on data.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0].values
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # There are invalid values in the related colum
    # Assuming that its a data labelling issue, an attempt is made to assume 2 as 1
    categories.loc[categories['related'] == 2, 'related'] = 1
    
    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep=False,inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    This function saves the cleaned and transformed data to
    the indicated database.

    Args:
        df: A pandas dataframe that contains clean data.
        database_filename: Path of the database file to be created.
    
    Returns:
        Nothing.
        Creates a database containing a table named 'figure-eight'
        with clean data.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('figure-eight', engine, index=False, if_exists = 'replace')


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