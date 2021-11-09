# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 22:15:46 2021

@author: G. Cao
"""

# So, Let’s get start. We will be using Boston House Pricing Dataset which is included 
# in the sklearn dataset API. We will load the dataset and separate out the features and targets.

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt 

boston = datasets.load_boston()
x = boston.data
y = boston.target
columns = boston.feature_names#create the dataframe
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.head()


# =============================================================================
# Discover outliers with visualization tools
# 
# Box plot-
# =============================================================================
import seaborn as sns
sns.boxplot(x=boston_df['DIS'])

# Above plot shows three points between 10 to 12, these are outliers as there are not 
# included in the box of other observation i.e no where near the quartiles.

# Here we analysed Uni-variate outlier i.e. we used DIS column only to check the outlier. 
# But we can do multivariate outlier analysis too. Can we do the multivariate analysis with 
# Box plot? Well it depends, if you have a categorical values then you can use that with any 
# continuous variable and do multivariate outlier analysis. As we do not have categorical value 
# in our Boston Housing dataset, we might need to forget about using box plot for multivariate outlier analysis.


# =============================================================================
# Scatter plot-
# =============================================================================

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['TAX'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()

# =============================================================================
# Discover outliers with mathematical function
# 
# =============================================================================
# Z-Score-

from scipy import stats
import numpy as np

z = np.abs(stats.zscore(boston_df))
print(z)

threshold = 3
print(np.where(z > 3))

# Don’t be confused by the results. The first array contains the list of row 
# numbers and second array respective column numbers, which mean z[55][1] have 
# a Z-score higher than 3.

print(z[55][1])


# So, the data point — 55th record on column ZN is an outlier.

# =============================================================================
# IQR score -
# =============================================================================
Q1 = boston_df.quantile(0.25)
Q3 = boston_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# As we now have the IQR scores, it’s time to get hold on outliers. The below code 
# will give an output with some true and false values. The data point where we have 
# False that means these values are valid whereas True indicates presence of an outlier.

print((boston_df < (Q1 - 1.5 * IQR)) |(boston_df > (Q3 + 1.5 * IQR)))


# =============================================================================
# Working with Outliers: Correcting, Removing
# =============================================================================

# Z-Score

# In the previous section, we saw how one can detect the outlier using Z-score but 
# now we want to remove or filter the outliers and get the clean data. This can be 
# done with just one line code as we have already calculated the Z-score.

boston_df_o = boston_df[(z < 3).all(axis=1)]
#Return whether all elements are True, potentially over an axis.
#axis=1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.
print(boston_df.shape)

print(boston_df_o.shape)
# So, above code removed around 90+ rows from the dataset i.e. outliers have been removed.


# =============================================================================
# IQR Score -
# =============================================================================

boston_df_out = boston_df[~((boston_df < (Q1 - 1.5 * IQR)) |(boston_df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(boston_df_out.shape)

# =============================================================================
# https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
# So the following in python (exp1 and exp2 are expressions which evaluate to a boolean result)...
# 
# exp1 and exp2              # Logical AND
# exp1 or exp2               # Logical OR
# not exp1                   # Logical NOT
# 
# ...will translate to...
# 
# exp1 & exp2                # Element-wise logical AND
# exp1 | exp2                # Element-wise logical OR
# ~exp1                      # Element-wise logical NOT
# 
# for pandas.
# 
# If in the process of performing logical operation you get a ValueError, then you need to use parentheses for grouping:
# 
# (exp1) op (exp2)
# 
# For example,
# 
# (df['col1'] == x) & (df['col2'] == y) 
# =============================================================================






























