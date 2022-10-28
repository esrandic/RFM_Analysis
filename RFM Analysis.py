#!/usr/bin/env python
# coding: utf-8

# In[1]:


########Customer Segmentation with RFM Analysis######


# In[2]:


#Business Problem


# In[3]:


#FLO, an online shoe store, wants to segment its customers and determine marketing strategies according to these segments. 
#To this end, 
#the behaviors of the customers will be defined and groups will be formed according to the clustering in these behaviors.


# In[4]:


#It consists of the information obtained from the past shopping behaviors of customers
#who made their last shopping from Flo as OmniChannel (both online and offline shopping) in the years 2020-2021.


# In[5]:


##The Story of Data Set
#master_id : Unique customer id,
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile),
# last_order_channel : The channel where the most recent purchase was made,
# first_order_date : Date of the customer's first purchase,
# last_order_date :Date of the customer's last purchase,
# last_order_date_online : The date of the last purchase made by the customer on the online platform,
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform,
# order_num_total_ever_online :The total number of purchases made by the customer on the online platform,
# order_num_total_ever_offline :The total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases,
# customer_value_total_ever_online : Total fee paid by the customer for online purchases,
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


# In[6]:


#Task 1: Understanding and Preparing the Data


# In[7]:


#Step1: Read the flo_data_20K.csv data. Make a copy of the dataframe.


# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv(r'C:\Users\esran\Desktop\DATA SET\flo_data_20k.csv')
df = df_.copy()


# In[9]:


#Step 2: In the dataset
# a. top 10 observations,


# In[10]:


df.head(10)


# In[11]:


# b. Variable names


# In[12]:


df.columns


# In[13]:


# c. descriptive statistics


# In[14]:


df.describe().T


# In[15]:


# d.Sum of Null values


# In[16]:


df.isnull().sum()


# In[17]:


# e.Variable types, review.


# In[18]:


df.dtypes


# In[19]:


# Step 3: Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for the total number of purchases and spending of each customer.


# In[20]:


df["total_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# In[21]:


# Step 4: Examine the variable types. Change the type of variables that express date to date.


# In[23]:


date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.dtypes


# In[24]:


# Step 5: Look at the distribution of the number of customers in the shopping channels, 
#the total number of products purchased and the total expenditures.


# In[25]:


df.groupby("order_channel").agg({"master_id":"count",
                                 "total_count":"sum",
                                 "total_price":"sum"})


# In[26]:


# Step6: List the top 10 customers with the highest earnings.


# In[27]:


df.sort_values("total_price", ascending=False)[:10]


# In[28]:


# Step 7: List the top 10 customers who ordered the most.


# In[29]:


df.sort_values("total_count", ascending=False)[:10]
df.groupby("master_id").agg({"total_count" : "sum"}).sort_values("total_count", ascending=False)[:10]


# In[30]:


# Step 8 : Functionalize all the data pre-process.


# In[31]:


def data_prep(dataframe):
    dataframe["total_count"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe


# In[32]:


data_prep(df)


# In[35]:


# Task 2: Calculating RFM Metrics


# In[36]:


df["last_order_date"].max()


# In[37]:


today_date = dt.datetime(2021, 6, 1)
type(today_date)


# In[38]:


df['total_count'].nunique()


# In[39]:


#Step 1: Make the definitions of Recency, Frequency and Monetary.


# In[ ]:


#"""recency: time since last purchase
#  frequency: total repeat purchases
#  monetary: average earnings per purchase"""


# In[ ]:


#Step 2: Calculate the Recency, Frequency and Monetary metrics for the customer.


# In[40]:


rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'total_count': lambda total_count: total_count,  
                                     'total_price': lambda total_price: total_price.sum()})


# In[41]:


rfm.head()


# In[43]:


#Step 3: Assign your calculated metrics to a variable named rfm.
#Step 4: Change the names of the metrics you created to recency, frequency, and monetary.


# In[44]:


rfm.columns = ['recency', 'frequency', 'monetary']


# In[45]:


#Task 3: Calculation of RF Score


# In[46]:


#Step 1: Convert the Recency, Frequency and Monetary metrics to scores between 1-5 with the help of qcut.


# In[47]:


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


# In[48]:


#Step 2: Record these scores as recency_score, frequency_score, and monetary_score. 
#Step 3: Express recency_score and frequency_score as a single variable and save as RF_SCORE.


# In[49]:


rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


# In[50]:


rfm.describe().T


# In[52]:


#Task 4: Defining RF Score by Segment


# In[53]:


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


# In[54]:


rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


# In[55]:


#Quest 5: Time for Action!


# In[ ]:


#Step1: Examine the recency, frequency and monetary averages of the segments.


# In[56]:


rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# In[ ]:


#Step2: With the help of RFM analysis, 
#find the customers in the relevant profile for the 2 cases given below and save the customer ids as csv.


# In[ ]:


#a.FLO includes a new brand of women's shoes in its structure. 
#The product prices of the brand it includes are above the general customer preferences.
#Therefore, it is desirable to contact the customers in the profile that will be interested in the promotion of the brand and the sales of the product.
#Customers to be contacted privately from loyal customers(champions,loyal_customers)
#and people who shop from the female category.
#Save these customers id numbers in the csv file


# In[69]:


rfm_ = pd.merge(df,rfm, on='master_id')


# In[70]:


rfm_.head()


# In[75]:


new = rfm_["new customers"] =rfm_.loc[((rfm_["segment"] == "loyal_customers") | (rfm_["segment"] == "champions")) & ((rfm_["interested_in_categories_12"].str.contains("KADIN")))]["master_id"]


# In[76]:


new.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)


# In[77]:


rfm_.head()


# In[78]:


# Action 2

# Planning %40 discount for male and child products. 
#With categories related to this discount people who are good customers 
#but not shopping for a long time and customers which we shouldnt lost , sleeping customers
#and new comers customers specially targeted. 
#Save the id numbers of these customers to the csv file.


# In[80]:


new2 = rfm_.loc[(rfm_["segment"] == "cant_loose") | (rfm_["segment"] == "hibernating") | (rfm_["segment"] == "new_customers") &
                  ((df["interested_in_categories_12"].str.contains("ERKEK") | (df["interested_in_categories_12"].str.contains("COCUK"))))]["master_id"]


# In[81]:


new2.to_csv("yeni_hedef_müşteri_id.csv", index=False)


# In[ ]:




