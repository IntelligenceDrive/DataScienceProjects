# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:57:08 2021

@author: G. Cao
"""
# https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

import pandas as pd
import numpy as np

# =============================================================================
# 1.Imputation
# =============================================================================

# The most simple solution to the missing values is to drop the rows or the entire column. 
# There is not an optimum threshold for dropping but you can use 70% as an example value and 
# try to drop the rows and columns which have missing values with higher than this threshold.


threshold = 0.7#Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]

# If you want to count the missing values in each column, try:

# df.isnull().sum() as default or df.isnull().sum(axis=0)

# On the other hand, you can count in each row (which is your question) by:

# df.isnull().sum(axis=1)


# =============================================================================
# Imputation is a more preferable option rather than dropping because it preserves the data size.
# =============================================================================

# Except for the case of having a default value for missing values, I think the best imputation 
# way is to use the medians of the columns. As the averages of the columns are sensitive to the outlier values, 
# while medians are more solid in this respect.


#Filling all missing values with 0
data = data.fillna(0)#Filling missing values with medians of the columns
data = data.fillna(data.median())



# =============================================================================
# Categorical Imputation
# =============================================================================

# Replacing the missing values with the maximum occurred value in a column is a 
# good option for handling categorical columns. But if you think the values in 
# the column are distributed uniformly and there is not a dominant value, imputing 
# a category like “Other” might be more sensible, because in such a case, your 
# imputation is likely to converge a random selection.

#Max fill function for categorical columns
data['column_name'].fillna(data['column_name'].value_counts()
.idxmax(), inplace=True)



# =============================================================================
# 2.Handling Outliers
# =============================================================================

# Outlier Detection with Standard Deviation


# If a value has a distance to the average higher than x * standard deviation, it 
# can be assumed as an outlier. Then what x should be?

# There is no trivial solution for x, but usually, a value between 2 and 4 seems practical.

#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = data['column'].mean () + data['column'].std () * factor
lower_lim = data['column'].mean () - data['column'].std () * factor

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]

# Outlier Detection with Percentiles

#Dropping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]

# An Outlier Dilemma: Drop or Cap

# Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
# On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.

#Capping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)data.loc[(df[column] > upper_lim),column] = upper_lim
data.loc[(df[column] < lower_lim),column] = lower_lim


# =============================================================================
# 3.Binning
# =============================================================================


# #Numerical Binning ExampleValue      Bin       
# 0-30   ->  Low       
# 31-70  ->  Mid       
# 71-100 ->  High

# #Categorical Binning ExampleValue      Bin       
# Spain  ->  Europe      
# Italy  ->  Europe       
# Chile  ->  South America
# Brazil ->  South America

# The main motivation of binning is to make the model more robust and prevent overfitting, 
# however, it has a cost to the performance. Every time you bin something, you sacrifice 
# information and make your data more regularized.

# The trade-off between performance and overfitting is the key point of the binning 
# process. In my opinion, for numerical columns, except for some obvious overfitting
# cases, binning might be redundant for some kind of algorithms, due to its effect 
# on model performance.

# However, for categorical columns, the labels with low frequencies probably affect 
# the robustness of statistical models negatively. Thus, assigning a general category 
# to these less frequent values helps to keep the robustness of the model. 
# For example, if your data size is 100,000 rows, it might be a good option to 
# unite the labels with a count less than 100 to a new category like “Other”.


#Numerical Binning Example
data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])   

# =============================================================================
# value   bin
# 0      2   Low
# 1     45   Mid
# 2      7   Low
# 3     85  High
# 4     28   Low
# 
# #Categorical Binning Example
# 
#      Country
# 0      Spain
# 1      Chile
# 2  Australia
# 3      Italy
# 4     Brazil
# 
# =============================================================================

conditions = [
    data['Country'].str.contains('Spain'),
    data['Country'].str.contains('Italy'),
    data['Country'].str.contains('Chile'),
    data['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')     

# =============================================================================
#      Country      Continent
# 0      Spain         Europe
# 1      Chile  South America
# 2  Australia          Other
# 3      Italy         Europe
# 4     Brazil  South America
# =============================================================================

# =============================================================================
# 4.Log Transform
# =============================================================================

 # 1. It helps to handle skewed data and after transformation, the distribution becomes 
 #    more approximate to normal.
 # 2. In most of the cases the magnitude order of the data changes within the range 
 #    of the data. For instance, the difference between ages 15 and 20 is not equal 
 #    to the ages 65 and 70. In terms of years, yes, they are identical, but for 
 #    all other aspects, 5 years of difference in young ages mean a higher magnitude 
 #    difference. This type of data comes from a multiplicative process and log transform 
 #    normalizes the magnitude differences like that.
 # 3. It also decreases the effect of the outliers, due to the normalization of 
 #    magnitude differences and the model become more robust.

#Log Transform Example
data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})
data['log+1'] = (data['value']+1).transform(np.log)

#Negative Values Handling
#Note that the values are different
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)   

# value  log(x+1)  log(x-min(x)+1)
# 0      2   1.09861          3.25810
# 1     45   3.82864          4.23411
# 2    -23       nan          0.00000
# 3     85   4.45435          4.69135
# 4     28   3.36730          3.95124
# 5      2   1.09861          3.25810
# 6     35   3.58352          4.07754
# 7    -12       nan          2.48491

# =============================================================================
# 5.One-hot encoding
# =============================================================================

# One-hot encoding is one of the most common encoding methods in machine learning. 
# This method spreads the values in a column to multiple flag columns and assigns 
# 0 or 1 to them. These binary values express the relationship between grouped and 
# encoded column.

# Why One-Hot?: If you have N distinct values in the column, it is enough to map 
# them to N-1 binary columns, because the missing value can be deducted from other 
# columns. If all the columns in our hand are equal to 0, the missing value must be 
# equal to 1. This is the reason why it is called as one-hot encoding. However, I 
# will give an example using the get_dummies function of Pandas. This function maps 
# all values in a column to multiple columns.

encoded_columns = pd.get_dummies(data['column'])
data = data.join(encoded_columns).drop('column', axis=1)

# =============================================================================
# 6.Grouping Operations
# =============================================================================

# Categorical Column Grouping

# The first option is to select the label with the highest frequency. 
# In other words, this is the max operation for categorical columns, 
# but ordinary max functions generally do not return this value, you 
# need to use a lambda function for this purpose.


data.groupby('id').agg(lambda x: x.value_counts().index[0])


# Second option is to make a pivot table. It can be defined as aggregated 
# functions for the values between grouped and encoded columns. 

#Pivot table Pandas Example
data.pivot_table(index='column_to_group', columns='column_to_encode', values='aggregation_column', aggfunc=np.sum, fill_value = 0)


# Last categorical grouping option is to apply a group by function after 
# applying one-hot encoding. This method preserves all the data -in the 
# first option you lose some-, and in addition, you transform the encoded 
# column from categorical to numerical in the meantime. You can check the 
# next section for the explanation of numerical column grouping.

# Numerical Column Grouping

#sum_cols: List of columns to sum
#mean_cols: List of columns to averagegrouped = data.groupby('column_to_group')

sums = grouped[sum_cols].sum().add_suffix('_sum')
avgs = grouped[mean_cols].mean().add_suffix('_avg')

new_df = pd.concat([sums, avgs], axis=1)


# =============================================================================
# 7.Feature Split
# =============================================================================
# data.name
# 0  Luther N. Gonzalez
# 1    Charles M. Young
# 2        Terry Lawson
# 3       Kristen White
# 4      Thomas Logsdon

#Extracting first names
data.name.str.split(" ").map(lambda x: x[0])
# 0     Luther
# 1    Charles
# 2      Terry
# 3    Kristen
# 4     Thomas

#Extracting last names
data.name.str.split(" ").map(lambda x: x[-1])
# 0    Gonzalez
# 1       Young
# 2      Lawson
# 3       White
# 4     Logsdon

# =============================================================================
# 8.Scaling
# =============================================================================

# Normalization
data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})

data['normalized'] = (data['value'] - data['value'].min()) / (data['value'].max() - data['value'].min())   value  normalized
# 0      2        0.23
# 1     45        0.63
# 2    -23        0.00
# 3     85        1.00
# 4     28        0.47
# 5      2        0.23
# 6     35        0.54
# 7    -12        0.10

# Standardization

data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})

data['standardized'] = (data['value'] - data['value'].mean()) / data['value'].std()

# =============================================================================
# 9.Extracting Date
# =============================================================================
 # Extracting the parts of the date into different columns: Year, month, day, etc.
 # Extracting the time period between the current date and columns in terms of years, months, days, etc.
 # Extracting some specific features from the date: Name of the weekday, Weekend or not, holiday or not, etc.

from datetime import date

data = pd.DataFrame({'date':
['01-01-2017',
'04-12-2008',
'23-06-1988',
'25-08-1999',
'20-02-1993',
]})

#Transform string to date
data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

#Extracting Year
data['year'] = data['date'].dt.year

#Extracting Month
data['month'] = data['date'].dt.month

#Extracting passed years since the date
data['passed_years'] = date.today().year - data['date'].dt.year

#Extracting passed months since the date
data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

#Extracting the weekday name of the date
data['day_name'] = data['date'].dt.day_name()        

#         date  year  month  passed_years  passed_months   day_name
# 0 2017-01-01  2017      1             2             26     Sunday
# 1 2008-12-04  2008     12            11            123   Thursday
# 2 1988-06-23  1988      6            31            369   Thursday
# 3 1999-08-25  1999      8            20            235  Wednesday
# 4 1993-02-20  1993      2            26            313   Saturday





