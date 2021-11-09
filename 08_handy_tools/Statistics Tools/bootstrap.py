#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# construct a population pickups for our lab
np.random.seed(42)
pickups = np.random.randint(0,500 , size=100)
pickups


# In[3]:


# population mean
pickups.mean()


# In[4]:


# population standard deviation
pickups.std()


# In[5]:


# draw a sample from population
sample = np.random.choice(pickups, size=30)
sample


# In[6]:


# our first sample mean
sample_mean = sample.mean()
sample_mean


# In[7]:


# standard deiveation for this sample
sample_std = np.std(sample, ddof=1)
sample_std


# In[8]:


# estimated standard error for sapmle mann
sample_std/(30 ** 0.5)


# In[9]:


# theorical standard error for sapmle mann
pickups.std()/(30 ** 0.5)


# In[10]:


# construct the simulated sampling distribution
sample_props = []
for _ in range(100000):
    sample = np.random.choice(pickups, size=30)
    sample_props.append(sample.mean())


# In[11]:


# the simulated mean of the sampling distribution
simulated_mean = np.mean(sample_props)


# In[12]:


# the simulated standard deviation of the sampling distribution
simulated_std = np.std(sample_props)


# In[13]:


# plot the simulated sampling distribution,
# under the Central Limit Theorem,
# it is expected normal
plt.hist(sample_props)


# In[14]:


# the theorical mean and simulated mean
(pickups.mean(), simulated_mean)


# In[15]:


# the theorical standard error and simulated standard error
(pickups.std()/(30 ** 0.5), simulated_std)




# =============================================================================
# Bootstrap Simulation
# =============================================================================

# draw a sample from population
sample = np.random.choice(pickups, size=30)
sample


# bootstrap for mean
boot_means = []
for _ in range(10000):
    bootsample = np.random.choice(sample,size=30, replace=True)
    boot_means.append(bootsample.mean())
    
    
# simulated mean of mean
bootmean = np.mean(boot_means)


# simulated standard deviation of mean
bootmean_std = np.std(boot_means)


# simulated mean VS true mean
(pickups.mean(), bootmean)


# the theorical standard error and simulated standard error
(pickups.std()/(30 ** 0.5), bootmean_std)






























