# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:02:34 2021

@author: G. Cao
"""

import pandas as pd
import numpy as np
import random

df_credit = pd.read_csv("data/train.csv")
df_credit


# =============================================================================
# Setting Up The Test Data
# =============================================================================

# To make the example make sense I am going to simplfy the “Home Ownership” feature 
# to have the two most common values and add a new feature called “Gender” with ~60% “Male” 
# and ~40% “Female” and then take a quick look at the results …

df_credit['Gender'] = np.random.choice(['Male', 'Female'], size=len(df_credit), p=[0.6, 0.4])

ownership_filter = df_credit['Home Ownership'].isin(['Home Mortgage', 'Rent'])
df_credit = df_credit.drop(df_credit[~ownership_filter].index)

print((df_credit['Home Ownership'].value_counts() / len(df_credit)).sort_values(ascending=False), 
      (df_credit['Gender'].value_counts() / len(df_credit)).sort_values(ascending=False))

# =============================================================================
# Preparing to Stratify
# =============================================================================

# The first thing we need to do is to create a single feature that contains all of 
# the data we want to stratify on as follows …
df_credit['Stratify'] = df_credit['Gender'] + ", " + df_credit['Home Ownership']
print((df_credit['Stratify'].value_counts() / len(df_credit)).sort_values(ascending=False))

# =============================================================================
# So there we have it, we have a set of proportions in our sample data that we intend to use to 
# train our model. However we check with our marketing team who assure us that the population 
# proportions are as follows …
# 
#     Male, Home Mortgage = 45% of the population
#     Male, Rent = 20% of the population
#     Female, Home Mortgage = 20% of the population
#     Female, Rent = 15% of the population
# =============================================================================
# =============================================================================
# Stratifying the Data
# =============================================================================


def stratify_data(df_data, stratify_column_name, stratify_values, stratify_proportions, random_state=None):
    """Stratifies data according to the values and proportions passed in
    Args:
        df_data (DataFrame): source data
        stratify_column_name (str): The name of the single column in the dataframe that holds the data values that will be used to stratify the data
        stratify_values (list of str): A list of all of the potential values for stratifying e.g. "Male, Graduate", "Male, Undergraduate", "Female, Graduate", "Female, Undergraduate"
        stratify_proportions (list of float): A list of numbers representing the desired propotions for stratifying e.g. 0.4, 0.4, 0.2, 0.2, The list values must add up to 1 and must match the number of values in stratify_values
        random_state (int, optional): sets the random_state. Defaults to None.
    Returns:
        DataFrame: a new dataframe based on df_data that has the new proportions represnting the desired strategy for stratifying
    """
    df_stratified = pd.DataFrame(columns = df_data.columns) # Create an empty DataFrame with column names matching df_data

    pos = -1
    for i in range(len(stratify_values)): # iterate over the stratify values (e.g. "Male, Undergraduate" etc.)
        pos += 1
        if pos == len(stratify_values) - 1: 
            ratio_len = len(df_data) - len(df_stratified) # if this is the final iteration make sure we calculate the number of values for the last set such that the return data has the same number of rows as the source data
        else:
            ratio_len = int(len(df_data) * stratify_proportions[i]) # Calculate the number of rows to match the desired proportion

        df_filtered = df_data[df_data[stratify_column_name] ==stratify_values[i]] # Filter the source data based on the currently selected stratify value
        df_temp = df_filtered.sample(replace=True, n=ratio_len, random_state=random_state) # Sample the filtered data using the calculated ratio
        
        df_stratified = pd.concat([df_stratified, df_temp]) # Add the sampled / stratified datasets together to produce the final result
        
    return df_stratified # Return the stratified, re-sampled data   




stratify_values = ['Male, Home Mortgage', 'Male, Rent', 'Female, Home Mortgage', 'Female, Rent']
stratify_proportions = [0.45, 0.20, 0.20, 0.15]
df_stratified = stratify_data(df_credit, 'Stratify', stratify_values, stratify_proportions, random_state=42)
df_stratified

# =============================================================================
# And just to be sure we have the right results, let’s take a look at the overall proportions of our Stratify 
# feature column ...
# =============================================================================

df_stratified.shape, df_credit.shape
# ((6841, 20), (6841, 20))

(df_stratified['Stratify'].value_counts() / len(df_stratified)).sort_values(ascending=False)

# Male, Home Mortgage      0.449934
# Female, Home Mortgage    0.199971
# Male, Rent               0.199971
# Female, Rent             0.150124
# Name: Stratify, dtype: float64



































