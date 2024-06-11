#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


# import the laibraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns",None)


# # Read Application csv

# In[2]:


app_data = pd.read_csv("C:\\Users\\defaultuser100000\\Desktop\\application_data.csv.zip")
app_data.head()


# # Data inspection on Application Dataset 

# # Get info and shape on the dataset 

# In[3]:


app_data.info()


# # Data Quality Check

# # Check for Percentage null values in Application dataset

# In[4]:


pd.set_option('display.max_rows',200)
app_data.isnull().mean()*100


# # Dropping Columns With Missing Greater than 47%

# In[5]:


percentage = 47 
threshold = int(((100-percentage)/100)*app_data.shape[0]+1)
app_df = app_data.dropna(axis = 1, how='any')
app_df = app_data.dropna(axis = 1,thresh = threshold)
app_df.head()


# In[6]:


app_df.isnull().mean()*100


# # Imput Missing Values  

# # Check The Missing Values in  Application Dataset Before Imputing

# In[7]:


app_df.info()


# # OCCUPATION _TYPE Column Has 31% Missing Values, Since Its a Categorical
# # Column, Imputing The Missing Values With a Unknown or Other Value

# In[8]:


app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[9]:


app_df.OCCUPATION_TYPE.value_counts(normalize = True)*100


# In[10]:


print(app_df["OCCUPATION_TYPE"].isna())
print("--------------")
print(app_df["OCCUPATION_TYPE"])
app_df["OCCUPATION_TYPE"].fillna("Others",inplace = True)


# In[11]:


app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[12]:


app_df.OCCUPATION_TYPE.value_counts(normalize = True)*100


# # EXT_SOURCE_3 Column has 19% Missing values

# In[13]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[14]:


app_df.EXT_SOURCE_3.value_counts(normalize = True)*100


# In[15]:


app_df.EXT_SOURCE_3.describe()


# In[16]:


sns.boxplot(app_df.EXT_SOURCE_3)
plt.show()


# In[17]:


app_df.EXT_SOURCE_3.fillna(app_df.EXT_SOURCE_3.median(),inplace = True)


# In[18]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[19]:


app_df.EXT_SOURCE_3.value_counts(normalize = True)*100


# In[20]:


null_cols = list(app_df.isna().any())
len(null_cols)


# In[21]:


app_df.isnull().mean()*100


# # Handling Missing Values in Columns With 13% Null Values

# In[22]:


app_df.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize = True)*100


# In[23]:


app_df.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize = True)*100


# - Conclusion : we counts see that 99% of values in the columns
#   AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,AMT_REQ_CREDIT_BUREAU_MON,
#   AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR is 0.0 . Hence imput these columns with the mode

# In[24]:


cols =["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR"]


# In[25]:


for col in cols:
    app_df[col].fillna(app_df[col].mode()[0],inplace = True)


# In[26]:


app_df.isnull().mean()*100


# # Handling Missing Values Less Than 1%

# In[27]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[28]:


app_df.NAME_TYPE_SUITE.value_counts(normalize = True)*100


# In[29]:


app_df.EXT_SOURCE_2.value_counts(normalize = True)*100


# In[30]:


app_df.OBS_30_CNT_SOCIAL_CIRCLE.value_counts(normalize = True)*100


# # . Conclusion
# ## . for categorical columns, imput the missing values with mode
# ## . for numerical columns, imput the missing values with median

# In[31]:


app_df.NAME_TYPE_SUITE.fillna(app_df.NAME_TYPE_SUITE.mode()[0],inplace = True)


# In[32]:


app_df.CNT_FAM_MEMBERS.fillna(app_df.CNT_FAM_MEMBERS.mode()[0],inplace = True)


# In[33]:


# IMPUTING NUMERICAL COLUMNS 
app_df.EXT_SOURCE_2.fillna(app_df.EXT_SOURCE_2.median(),inplace = True)
app_df.AMT_GOODS_PRICE.fillna(app_df.AMT_GOODS_PRICE.median(),inplace = True)
app_df.AMT_ANNUITY.fillna(app_df.AMT_ANNUITY.median(),inplace = True)
app_df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_60_CNT_SOCIAL_CIRCLE.median(),inplace = True)
app_df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_30_CNT_SOCIAL_CIRCLE.median(),inplace = True)
app_df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_30_CNT_SOCIAL_CIRCLE.median(),inplace = True)
app_df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_60_CNT_SOCIAL_CIRCLE.median(),inplace = True)
app_df.DAYS_LAST_PHONE_CHANGE.fillna(app_df.DAYS_LAST_PHONE_CHANGE.median(),inplace = True)


# In[34]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[35]:


app_df.isnull().mean()*100


# In[36]:


null_cols=list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[37]:


app_df.isnull().mean()*100


# # Convert Nagative Vlues To Positive In Days Variables So That Median Is Not Affected

# In[38]:


app_df.DAYS_BIRTH = app_df.DAYS_BIRTH.apply(lambda x: abs(x))
app_df.DAYS_EMPLOYED = app_df.DAYS_EMPLOYED.apply(lambda x: abs(x))
app_df.DAYS_ID_PUBLISH = app_df.DAYS_ID_PUBLISH.apply(lambda x: abs(x)) 
app_df.DAYS_LAST_PHONE_CHANGE = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: abs(x))
app_df.DAYS_REGISTRATION = app_df.DAYS_REGISTRATION.apply(lambda x: abs(x))        


# # Binning of Continuous Variables
# ## Standardizing Days COlumns in Year For Easy Binning

# In[39]:


app_df["YEARS_BIRTH"] = app_df.DAYS_BIRTH.apply(lambda x: int(x//356))
app_df["YEARS_EMPLOYED"] = app_df.DAYS_EMPLOYED.apply(lambda x:int(x//356))
app_df["YEARS_ID_PUBLISH"] = app_df.DAYS_ID_PUBLISH.apply(lambda x: int(x//356)) 
app_df["YEARS_BIRTH"] = app_df.DAYS_BIRTH.apply(lambda x: int(x//356))
app_df["YEARS_LAST_PHONE_CHANGE"] = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: int(x//356))
app_df["YEARS_REGISTRATION"] = app_df.DAYS_REGISTRATION.apply(lambda x: int(x//356)) 


# # Binning AMT_CREDIT Columns

# In[40]:


app_df.AMT_CREDIT.value_counts(normalize = True)*100


# In[41]:


app_df.AMT_CREDIT.describe()


# In[42]:


app_df["AMT_CREDIT_Category"] = pd.cut(app_df.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000],
                                    labels = ["Very low Credit","Low Credit","Medium Credit","High Credit","Very High Credit"])
print(app_df.columns)


# In[43]:


app_df.AMT_CREDIT_Category.value_counts(normalize = True)*100


# In[44]:


app_df["AMT_CREDIT_Category"].value_counts(normalize = True).plot.bar()
plt.show()


# - Conclusion The Credit Amount of The Loan  For Amount Low(2L to 4L)or Very High (above 8L)
#   

#    ###     Binning YEARS_BIRTH Column

# In[45]:


app_df["AGE_Category"] = pd.cut(app_df.YEARS_BIRTH, [0,25,45,65,85],
                                labels = ["Below-25","25-45","45-65","65-85"])


# In[46]:


app_df.AGE_Category.value_counts(normalize=True)*100


# In[47]:


app_df["AMT_CREDIT_Category"].value_counts(normalize = True).plot.pie(autopct='%1.2f%%')
plt.show()


# - Conclusion Most of the Application are between 25-45 age Group

#    #    Data Imbalance Check

# In[48]:


app_df.head()


# # Diving Application Dataset With Target Variable as 0 and 1

# In[49]:


tar_0 = app_df[app_df.TARGET==0]
tar_1 = app_df[app_df.TARGET==1]


# In[50]:


app_df.TARGET.value_counts(normalize = True)*100


# - Conclusion 1 out of 9/10 applicant are defauls

# # Univariate Analysis

# In[51]:


cat_cols = list(app_df.columns[app_df.dtypes==object])
num_cols = list(app_df.columns[app_df.dtypes==np.int64]) + list(app_df.columns[app_df.dtypes==np.float64])


# In[52]:


cat_cols


# In[53]:


num_cols


# In[54]:


for col in cat_cols:
    print(app_df[col].value_counts(normalize = True))
    plt.figure(figsize=[5,5])
    app_df[col].value_counts(normalize = True).plot.pie(labeldistance=None, autopct = '%1.2f%%')
    plt.legend()
    plt.show()


# - conclusion>>insights on below columns
# 
# 1. NAME_CONTRACT_TYPE-More applicants have Cash loans than Reciving loans
# 2. CODE_GENDER-Number of Female applicants are twice than that of male applicants
# 3. FLAG_OUN_CAR-Most(70%)of the applicants do not own a car
# 4. FLAG_OWN_REALTY-mOST(70%) of the applicants do not own a house 
# 5. NAME_TYPE_SUITE-More(81%)of the applicants are Unaccompanied
# 6. NAME_INCOME_TYPE-More(51%)of the applicants are earning their income frome work
# 7. NAME_EDUCATION_TYPE-(71%)of the applicants have completed Secondary/secondary education
# 8. NAME_FAMILY_STATUS-(63%) of the applicants are married
# 9. NAME_HOUSING_TYPE-(88%)of the housing type of applicants are House/apartment
# 10. OCCUPATION_TYPE_MOST-(31%)of the applicants have other occupation type
# 11. WEEKDAY_APPR_PROCESS_START-Most of the applicants have applied the loan on tuesday
# 12. ORGANIZATION_TYPE-Most of the organization type of employees are business Entity type 3

# # Plot on Numerical Columns
# ## Categorizing Columns with or without Flags

# In[55]:


num_cols_withoutflag = []
num_cols_withflag = []
for col in num_cols:
    if col.startswith("FLAG"):
        num_cols_withflag.append(col)
    else:
            num_cols_withoutflag.append(col)


# In[56]:


num_cols_withflag


# In[57]:


num_cols_withoutflag


# In[58]:


for col in num_cols_withoutflag:
    print(app_df[col].describe())
    plt.figure(figsize=[8,5])
    sns.boxplot(data=app_df,x=col)
    plt.show()
    print("------------------------")


# - conclusion>>Few Columns are with outliers are below
# 
# 1. AMT_INCOME_TOTAL Column has a few outliers and there is a huge difference between the 99th percentile and the max value,also we could see huge variation in mean and median due to outliers
# 2. AMT_CREDIT Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see huge variation in mean and median due to outliers
# 3. AMT_ANNUITY Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 4. AMT_GOODS_PRICE Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 5. REGION_POPULATION_RELATIVE Column has a one outliers and there not much difference between mean and median
# 
# # Univariate Analisis on columns with Targert 0 and 1

# In[59]:


for col in cat_cols:
    print(f"plot on {col} for Target 0 and 1")
    plt.figure(figsize=[10,7])
    plt.subplot(1,2,1)
    tar_0[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    tar_1[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
    print("----------------")


# - conclusion>>below are the column insights
# 
# 1. NAME_CONTRACT_TYPE-More application have Cash loans than Reciving loans
# 2. CODE_GENDER-Number of Female applicants are twice than that of male applicants
# 3. FLAG_OUN_CAR-Most(70%)of the applicants do not own a car
# 4. FLAG_OWN_REALTY-mOST(70%) of the applicants do not own a house 
# 5. NAME_TYPE_SUITE-More(81%)of the applicants are Unaccompanied
# 6. AME_INCOME_TYPE-More(51%)of the applicants are earning their income frome work
# 7. AME_EDUCATION_TYPE-(71%)of the applicants have completed Secondary/secondary education
# 8. AME_FAMILY_STATUS-(63%) of the applicants are married
# 9. AME_HOUSING_TYPE-(88%)of the housing type of applicants are House/apartment
# 10. OCCUPATION_TYPE_MOST-(31%)of the applicants have other occupation type
# 11. WEEKDAY_APPR_PROCESS_START-Most of the applicants have applied the loan on tuesday
# 12. ORGANIZATION_TYPE-Most of the organization type of employees are business Entity type 3

# # Analysis on AMT_GOODS_PRICE on Target 0 and 1

# In[60]:


plt.figure(figsize=[10,6])
sns.distplot(tar_0['AMT_GOODS_PRICE'],label='tar_0',hist=False)
sns.distplot(tar_1['AMT_GOODS_PRICE'],label='tar_1',hist=False)
plt.legend()
plt.show()


# - conclusion the price goods for which loan is given has the same varation for Taget 0 and 1

# # Bivariate and Multivariate Analysis
# ## Bivariate Analysis between WEEKDAY_APPR_PROCESS_START vs HOUR_APPR_PROCESS_START

# In[61]:


plt.figure(figsize=[15,10])
plt.subplot(1,2,1)
sns.boxplot(x="WEEKDAY_APPR_PROCESS_START",y="HOUR_APPR_PROCESS_START",data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x="WEEKDAY_APPR_PROCESS_START",y="HOUR_APPR_PROCESS_START",data=tar_1)
plt.show()


# - Conclusion>>
# 
# 1.The Bank operates between 10am to 3pm except from saturday and sunday,its between 10am to 2pm.
# 
# 2.We can observe that around 11.30am to 12pm around 50% of customers visit the branch for loan application on all the days except for saturday where the time is between 10am to 11am for both target 0 and 1.
# 
# 3.The loan defaulters have applied for the loan between 9.30am-10am and 2pm where as the applicants who repay the loan on time have applied for the loan between 10am to 3pm.

# # Bivariate Analysis between AGE_CATEGORY vs AMT_CREDIT

# In[62]:


plt.figure(figsize=[15,10])

plt.subplot(1,2,1)
sns.boxplot(x='AGE_Category', y = 'AMT_CREDIT', data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='AGE_Category', y = 'AMT_CREDIT', data=tar_1)
plt.show()


# - Conclusion>>
# 
# 1. The applicants between age group 25 to 65 have credit amount of the loan less than 2500000 and are abel to repay the loan properly
# 2. The applicants with less than 100000 Credit amount are with age group greater than 65 may be consider as defaulters
# 3. Most applicants who have credit amount of the loan less than 1700000 are loan defaulters with 25 and less than age
# 
# # Pair Plot of Amount Columns for Target 0

# In[63]:


sns.pairplot(tar_0[["AMT_INCOME_TOTAL","AMT_ANNUITY","AMT_GOODS_PRICE"]])
plt.show()


# - Conclusion>>
# 
# 1. AMT_CREDIT increases or varies linearly with AMT_GOODS_PRICE and AMT_CREDIT increases with AMT_ANNUITY
# 2. AMT_ANNUITY increases with increases in AMT_GOODS_PRICE and AMT_CREDIT
# 3. AMT_GOODS_PRICE increases with increases in AMT_CREDIT and AMT_ANNUITY
# 4. AMT_INCOME_TOTAL has a drastic increases with slight increases in AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE
# 
# # Pair Plot of Amount Columns for Target 1

# In[64]:


sns.pairplot(tar_1[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]])
plt.show()


# - Conclusion>>For Applicants who are unable to replay the loan on time
# 
# 1. AMT_CREDIT increases or varies linearly with AMT_GOODS_PRICE and AMT_CREDIT increases with AMT_ANNUITY
# 2. AMT_ANNUITY increases with increases in AMT_GOODS_PRICE and AMT_CREDIT
# 3. AMT_GOODS_PRICE increases with increases in AMT_CREDIT and AMT_ANNUITY
# 4. AMT_INCOME_TOTAL has a drastic increases wuth slight increases in AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE
# 
# # Co-relation between Numerical Columns

# In[65]:


corr_data=app_df[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"
                     ,"YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data.head()


# In[66]:


corr_data.corr()


# In[67]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_data.corr(), annot=True,cmap="RdYlGn")
plt.show()


#  - Conclusion>>
#  
# 1. AMT_INCOME_TOTAL has a positive corelation index of 0.16,0.19,0.16 with AMT_CREDIT,AMT_ANNUITY, AMT_GOODS_PRICE respectively.
# 2. AMT_CREDIT has negative corelation index of 0.64 with YEARS_EMPLOYED and positive corelation index of 0.99,0.77 with AMT_GOODS_PRICE,AMT_ANNUITY respectively.
# 3. AMT_ANNUITY has negative corelation index of 0.1 with YEARS_EMPLOYED and positive corelation index of 0.77 with AMT_CREDIT
# 4. AMT_GOODS_PRICE has a positive corelation with AMT_CREDIT ,AMT_ANNUITY
# 5. YEARS_BIRTH has a positive corelation with YEARS_EMPLOYED,AMT_GOODS_PRICE negative corelation with AMT_ANNUITY,AMT_INCOME_TOTAL
# 6. YEARS_EMPLOYED has negative corelation index of 0.1 with AMT_ANNUITY and has a positive corelation with YEARS_REGISTRATION,YEARS_ID_PUBLISH 
# 7. YEARS_REGISTRATION has a positive corelation index with YEARS_ID_PUBLISH,YEARS_BIRTH,YEARS_EMPLOYED
# 8. YEARS_ID_PUBLISH has a positive corelation with YEARS_REGISTRATION and negative corelation with  AMT_INCOME_TOTAL,AMT_ANNUITY
# 9. YEARS_LAST_PHONE_CHANGE has negative corelation with YEARS_EMPLOYED and positive corelation with AMT_GOODS_PRICE,YEARS_ID_PUBLISH
# 
# 
# # Split Numerical variables based on Target 0 and 1 to find Co-relation

# In[68]:


corr_data_0=app_df[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"
                     ,"YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_0.head()


# In[69]:


corr_data_1=app_df[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"
                     ,"YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_1.head()


# In[70]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_data_0.corr(), annot=True,cmap="RdYlGn")
plt.show()


# - Conclusion>>
# 
# 1. AMT_INCOME_TOTAL has a positive corelation index of 0.34,0.42,0.35 with AMT_CREDIT,AMT_ANNUITY, AMT_GOODS_PRICE respectively and Negative with most of the other Year columns
# 2. AMT_CREDIT has a strong positive corelation index of 0.99,0.77 with AMT_GOODS_PRICE and AMT_ANNUITY
# 3. AMT_ANNUITY has positive corelation index of 0.77,0.78 with AMT_CREDIT,AMT_GOODS_PRICE and Negative with most of the other Year columns
# 4. AMT_GOODS_PRICE has a positive corelation index 0.78,0.99 with AMT_CREDIT ,AMT_ANNUITY

# In[71]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_data_1.corr(), annot=True, cmap="RdYlGn")
plt.show()


# - Conclusion>>
# 
# 1. AMT_INCOME_TOTAL has a less corelation with AMT_CREDIT,AMT_ANNUITY, AMT_GOODS_PRICE respectively 
# 2. AMT_CREDIT has a strong positive corelation index of 0.98,0.75 with AMT_GOODS_PRICE and AMT_ANNUITY resp. and also positive with other Year columns
# 3. AMT_ANNUITY has positive corelation index of 0.75 with AMT_CREDIT ,AMT_GOODS_PRICE and Negative with YEAR_EMPLOED,YEAR_REGISTRATION
# 4. AMT_GOODS_PRICE has a positive corelation index 0.75,0.98 with AMT_CREDIT ,AMT_ANNUITY AND week positive corelation with other year column
# 
# # Read Previous Application CSV

# In[72]:


papp_data = pd.read_csv("C:\\Users\\mohit\\Downloads\\previous_application.csv (1).zip")
papp_data.head()


# # Data Inspection on Previous Application dataset
# 
# ## Get info and shape on the dataset

# In[74]:


papp_data.info()


# In[75]:


papp_data.shape


# # Data Quality Check
# 
# ## check for Percentage Null Values in Application dataset

# In[77]:


papp_data.isnull().mean()*100


# In[76]:


percentge = 49 
threshold = int(((100-percentage)/100)*papp_data.shape[0]+1)
papp_df = papp_data.dropna(axis = 1, how='any')
papp_df = papp_data.dropna(axis = 1,thresh = threshold)
papp_df.head()


# In[78]:


papp_df.shape


# # Impute Missing Values
# ## Check the dtype of Missing Values in Application dataset Before Imputing Values

# In[80]:


# papp_df.DAYS_FIRST_DRAWING = papp_df.DAYS_FIRST_DRAWING.apply(lambda x: abs(x))
# papp_df.DAYS_FIRST_DUE = papp_df.DAYS_FIRST_DUE.apply(lambda x: abs(x))
# papp_df.DAYS_LAST_DUE_1ST_VERSION = papp_df.DAYS_LAST_DUE_1ST_VERSION.apply(lambda x: abs(x)) 
# papp_df.DAYS_LAST_DUE = papp_df.DAYS_LAST_DUE.apply(lambda x: abs(x))
# papp_df.DAYS_TERMINATION = papp_df.DAYS_TERMINATION.apply(lambda x: abs(x)
# papp_df.DAYS_DECISION = papp_df.DAYS_DECISION.apply(lambda x: abs(x)                                                          


# In[79]:


for col in papp_df.columns:
    if papp_df[col].dtypes == np.int64 or papp_df[col].dtypes == np.float64:
        papp_df[col] = papp_df[col].apply(lambda x: abs(x))


# # Validate if any null values present in dataset

# In[82]:


null_cols = list(papp_df.columns[papp_df.isna().any()])
len(null_cols)


# In[81]:


papp_df.isnull().mean()*100


# # Binning of Continuous Variables
# ## Binning AMT_CREDIT Column

# In[83]:


papp_df.AMT_CREDIT.describe()


# In[84]:


papp_df["AMT_CREDIT_Category"] = pd.cut(papp_df.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000],
                                    labels = ["Very low Credit","Low Credit","Medium Credit","High Credit","Very High Credit"])


# In[85]:


papp_df.AMT_CREDIT_Category.value_counts(normalize=True)*100


# In[86]:


papp_df["AMT_CREDIT_Category"].value_counts(normalize = True).plot.bar()
plt.show()


# - Conclusion>> The Credit Amount of The Loan  For most applicants is either low(200000 to 400000)

# In[87]:


print(papp_df["AMT_GOODS_PRICE"])
papp_df["AMT_GOODS_PRICE_Category"] = pd.cut(x = papp_df["AMT_GOODS_PRICE"], bins = [0,200000,400000,600000,800000,1000000], 
                                        labels = ["Very-low Price","Low Price","Medium Price","High Price","Very High Price"])


# In[88]:


papp_df.isnull().mean()*100


# In[89]:


null_cols=list(papp_df.columns[papp_df.isna().any()])
len(null_cols)


# In[90]:


print(papp_df.AMT_GOODS_PRICE_Category)
papp_df["AMT_GOODS_PRICE_Category"].value_counts(normalize = True).plot.pie(autopct='%1.2f%%')
plt.legend()
plt.show()


# # Data Imbalance Check
# ## Dividing Application Dataset with NAME_CONTTRACT_STATUS

# In[91]:


approved = papp_df[papp_df.NAME_CONTRACT_STATUS == "Approved"]
cancelled = papp_df[papp_df.NAME_CONTRACT_STATUS == "Cancelled "]
refused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Refused"]
unused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Unused"]


# In[92]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize = True)*100


# In[93]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize = True).plot.pie(labeldistance=None, autopct = '%1.2f%%')
plt.legend()
plt.show()


# - Conclusion>> 62% of the Applicants have the loan approved 19%, 17% Applicants are Rejected or Cancelled amd 2% are unused

# # Univariate Analisis

# In[94]:


cat_cols = list(papp_df.columns[papp_df.dtypes==object])
num_cols = list(papp_df.columns[papp_df.dtypes==np.int64]) + list(papp_df.columns[papp_df.dtypes==np.float64])


# In[95]:


cat_cols


# In[96]:


num_cols


# In[97]:


cat_cols = ['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION']


# In[98]:


num_cols = ['SK_ID_PREV','SK_ID_CURR','HOUR_APPR_PROCESS_START','NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','SELLERPLACE_AREA','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','CNT_PAYMENT','DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL']


# # Plot Categorical Columns 

# In[99]:


for col in cat_cols:
    print(papp_df[col].value_counts(normalize = True)*100)
    plt.figure(figsize=[5,5])
    papp_df[col].value_counts(normalize = True).plot.pie(labeldistance=None, autopct = '%1.2f%%')
    plt.legend()
    plt.show()
    print('------------------------')


# - conclusion>>insights on below columns
# 
# 1. NAME_CONTRACT_TYPE-45% applicants Recived cash loans 44% Applicants Recived Conumer loans Revolving during previous
# 2. WEEKDAY_APPR_PROCESS_START-All the days have almost equal number of previous loan application
# 3. NAME_CONTRACT_STATUS-More(62%)of the applicants are approved 19% Cancelled, 17% Refused and 2% unused
# 4. NAME_PAYMENT_TYPE-More(62%)of payment type are cash through bank  32% Other modes 
# 5. NAME_CLIENT_TYPE-More(74%)of the applicants are Rpeaters,18% are new applicants 8% are refreshed applicants
# 6. NAME_SELLER_INDUSTRY-(51%) Are from industries, 24%,17% are from Consumer electronics, Connectivity Industry respectively
# 7. CHANNEL_TYPE-43% Channel type is Credit and Cash offices,29% are Country wide
# 8. NAME_YIELD_GROUP-Majority of the yield group are others
# 9. PRODUCT_COMBINATION-Most used PRODUCT_COMBINATION is Cash followed by POS household with interest, POS mobile with interest

# ## Plot on Numerical Columns

# In[100]:


for col in num_cols:
    print("99th Percentile",np.percentile(papp_df[col],99))
    print(papp_df[col].describe())
    plt.figure(figsize=[10,6])
    sns.boxplot(data=papp_df,x=col)
    plt.show()
    print("---------------")


# - conclusion>>Few Columns are with outliers are below
# 
# 1. HOUR_APPR_PROCESS_START Column has a few outliers and small variation in mean and median due to outliers
# 2. AMT_CREDIT Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see huge variation in mean and median due to outliers
# 3. AMT_ANNUITY Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 4. AMT_GOODS_PRICE Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 5. AMT_APPLICATION Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see huge variation in mean and median due to outliers
# 6. CNT_PAYMENT Column has a few outliers and there is a small difference between mean and median
# 7. DAYS_DECISION Column has a few outliers and there small difference between mean and median

# # Bivariate and Multivariate Analysis
# ## Bivariate Analysis between WEEKDAY_APPR_PROCESS_START vs AMT_APPLICATION

# In[101]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=approved)
plt.title("Plot for Approved")
plt.show()


# In[102]:


cancelled = papp_df[papp_df.NAME_CONTRACT_STATUS == "cancelled"]
cancelled = papp_df


# In[103]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=cancelled)
plt.title("Plot for Cancelled")
plt.show()


# In[104]:


unused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Unused"]
unused = papp_df


# In[105]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=unused)
plt.title("Plot for Unused")
plt.show()


# In[106]:


plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=refused)
plt.title("Plot for refused")
plt.show()


# - Conclusion >>
# 
# 1. The Credit Amount of Applicants with approved status is high on monday and wednesday than other days and least on sunday
# 2. The Credit Amount of Applicants with cancelled status is high on Sunday and almost equal on other days 
# 3. The Credit Amount of Applicants with refused status is least on Sunday and more on Monday and wednesday
# 4. The Credit Amount of Applicants with unsed offer status is almost equal on all days

# In[107]:


plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Approved")
sns.scatterplot (x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=approved)
plt.subplot(1,4,2)
plt.title("cancelled")
sns.scatterplot (x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=cancelled)
plt.subplot(1,4,3)
plt.title("refused")
sns.scatterplot (x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=refused)
plt.subplot(1,4,4)
plt.title("unused")
sns.scatterplot (x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=unused)
plt.show()


# - Conclusion >>
# 
# 1. For loan status as Approved Refused Cancelled Amount of annuity increases with goods price
# 2. For loan status as Refused it has no linear relationship
# 
# # Co-relation between Numerical Columns

# In[108]:


corr_approved = approved[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_refused = refused[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_cancelled = cancelled[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_unused = unused[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_approved


# ## Co-relation for Numerical columns for Approved

# In[109]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_approved.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Approved")
plt.show()


# - Conclusion >>
# 
# 1. AMT_APPLICATION has higher Corelation with AMT_CREDIT and AMT_GOODS_PRICE,AMT_ANNUITY
# 2. DAYS_DECISION has nagetive corelation with AMT_GOODS_PRICE,AMT_CREDIT,AMT_APPLICATION,CNT_PAYMENT,AMT_ANNUITY
# 
# # Co-relation for Numerical columns for Refused

# In[110]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_refused.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Refused")
plt.show()


# - Conclusion >>
# 
# 1. AMT_APPLICATION has higher Corelation with AMT_CREDIT and AMT_GOODS_PRICE,AMT_ANNUITY
# 2. DAYS_DECISION has nagetive corelation with AMT_GOODS_PRICE,AMT_CREDIT,AMT_APPLICATION,CNT_PAYMENT,AMT_ANNUITY
# 
# # Co-relation for Numerical columns for Cancelled

# In[111]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_cancelled.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Cancelled")
plt.show()


# - Conclusion >>
# 
# 1. AMT_APPLICATION has higher Corelation with AMT_CREDIT and AMT_GOODS_PRICE,AMT_ANNUITY
# 2. DAYS_DECISION has nagetive corelation with AMT_GOODS_PRICE,AMT_CREDIT,AMT_APPLICATION,CNT_PAYMENT,AMT_ANNUITY
# 
# 
# # Co-relation for Numerical columns for Unused

# In[112]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_unused.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Unused")
plt.show()


# - Conclusion >>
# 
# 1. AMT_APPLICATION has higher Corelation with AMT_CREDIT and AMT_GOODS_PRICE
# 2. DAYS_DECISION and CNT_PAYMENT has nagetive corelation with AMT_GOODS_PRICE,AMT_CREDIT,AMT_APPLICATION,CNT_PAYMENT
# 

# # Merge the Application and Previous Application DataFrame

# In[113]:


merge_df = app_df.merge(papp_df,on=["SK_ID_CURR"],how = 'left')
merge_df.head()


# In[114]:


merge_df.info()


# #  Filtering required columns for our analysis 

# In[115]:


for col in merge_df.columns:
    if col.startswith("FLAG"):
        merge_df.drop(columns=col,axis=1,inplace=True)
merge_df.shape


# In[116]:


merge_df.shape


# In[117]:


res1=pd.pivot_table(data=merge_df,index=["NAME_INCOME_TYPE","NAME_CLIENT_TYPE"],
                   columns=["NAME_CONTRACT_STATUS"],
                   values="TARGET",aggfunc="mean")
res1


# In[118]:


plt.figure(figsize=[10,10])
sns.heatmap(res1,annot=True,cmap="BuPu")
plt.show()


# - Conclusion>>
# 
# 1. Applicants with income type Maternity leave and client type New are having more chances of getting the loan approved
# 2. Applicants with income type Maternity leave,Unemployed and client type Repeater are having more chances of getting the loan cancelled
# 3. Applicants with income type Maternity leave,Unemployed and client type Repeater are having more chances of getting the loan Refused
# 4. Applicants with income type Maternity leave and client type Repeater,Working and client type New are not able to utilize the Banker's offer

# In[119]:


res2=pd.pivot_table(data=merge_df,index=["CODE_GENDER","NAME_SELLER_INDUSTRY"],
                   columns=["TARGET"],values="AMT_GOODS_PRICE_x",aggfunc="sum")
res2


# In[121]:


plt.figure(figsize=[10,10])
sns.heatmap(res2,annot=True,cmap="BuPu")
plt.show()

