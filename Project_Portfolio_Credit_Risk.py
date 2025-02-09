#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as mp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[10]:


credit_risk = pd.read_csv(r"D:\Excel_Campus\credit_risk_dataset.csv")
credit_risk.head(5)


# In[11]:


credit_risk.shape


# In[12]:


credit_risk.describe


# In[13]:


credit_risk.describe()


# In[14]:


credit_risk.tail()


# In[15]:


credit_risk_copy = credit_risk.copy()


# In[16]:


credit_risk_copy


# In[17]:


credit_risk.pivot_table(index= "person_age", values= "person_income", aggfunc='count')


# In[18]:


cr_age_removed = credit_risk[credit_risk['person_age']<=70]
# removing ages greater than 70


# In[19]:


cr_age_removed


# In[20]:


cr_age_removed.pivot_table(index ='person_age', values ="person_income", aggfunc= "count")


# In[21]:


#checking employment length
cr_age_removed.pivot_table(
    
    index='person_emp_length', 
    columns='loan_status', 
    values='person_income', 
    aggfunc='count'
    
).reset_index().sort_values(by="person_emp_length", ascending=False)


# In[22]:


#removing employment more than 47 years

person_emp_removed=cr_age_removed[cr_age_removed['person_emp_length']<=47]


# In[23]:


person_emp_removed.shape


# In[24]:


person_emp_removed.describe()


# In[25]:


#checking for null Values
person_emp_removed.isnull().sum()


# In[26]:


cr_data= person_emp_removed.copy()
cr_data


# In[27]:


person_emp_removed.head()
person_emp_removed.reset_index(drop = True , inplace=True)
person_emp_removed


# In[28]:


# filling null values in loan inetrest where ther is 0%  with median
cr_data_ok = cr_data.fillna({"loan_int_rate":cr_data["loan_int_rate"].median()})


# In[29]:


cr_data_ok.isnull().sum()
cr_data.reset_index(drop = True , inplace=True)
cr_data


# In[30]:


cr_data_ok.describe()
cr_data_ok.reset_index(drop= True,inplace= True)
cr_data_ok


# # Working on Categorical Features

# In[31]:


cr_data_ok.head()


# In[32]:


cr_data_copy= cr_data_ok.copy()


# In[33]:


cr_data_copy


# In[34]:


cr_data_copy.groupby(cr_data_copy['person_home_ownership']).count()['person_age']


# In[68]:


cr_ownership = pd.get_dummies(cr_data_copy['person_home_ownership'],drop_first=True).astype("int")


# In[36]:


cr_ownership


# In[37]:


pd.get_dummies(cr_data_copy['loan_intent'],drop_first=True).astype("int")
cr_loan_intent =pd.get_dummies(cr_data_copy['loan_intent'],drop_first=True).astype("int")


# In[38]:


cr_loan_intent


# In[39]:


import numpy as np
cr_data_copy['cb_person_default_on_file_binary'] = np.where(cr_data_copy['cb_person_default_on_file'] =="Y",1 ,0)


# In[40]:


cr_data_copy


# In[41]:


# data scaling
cr_data_scale = cr_data_copy.drop(['loan_status','loan_grade','loan_intent','person_home_ownership','cb_person_default_on_file','cb_person_default_on_file_binary'],axis=1)


# In[42]:


cr_data_scale


# In[43]:


scaler = StandardScaler()


# In[44]:


#uniform distribution: mean 0, ste dev=1
scaled_data = scaler.fit_transform(cr_data_scale)


# In[45]:


scaled_data


# In[46]:


scaled_data.shape



# In[47]:


scaled_df = pd.DataFrame(scaled_data)


# In[48]:


scaled_df


# In[49]:


cr_data_scale.columns


# In[50]:


scaled_df = pd.DataFrame(scaled_data,columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'])


# In[51]:


scaled_df


# In[52]:


round(np.std(scaled_df.person_age),2)


# In[53]:


round(np.mean(scaled_df.person_age),2)


# In[54]:


scaled_df_combined = pd.concat([scaled_df,cr_ownership,cr_loan_intent],axis=1)


# In[55]:


scaled_df_combined.shape


# In[78]:


scaled_df_combined


# In[79]:


scaled_df_combined['cb_person_default_on_file'] = cr_data_copy['cb_person_default_on_file_binary']
scaled_df_combined['loan_status'] = cr_data_copy['loan_status']

scaled_df_combined.head()


# In[80]:


pip install imbalanced-Learn


# In[81]:


pip install imbalanced-Learn


# In[82]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[88]:


target = scaled_df_combined['loan_status']
features = scaled_df_combined.drop('loan_status',axis=1)


# In[89]:


features.head()


# In[90]:


balanced_features, balanced_target = smote.fit_resample(features,target)


# In[91]:


balanced_features.shape


# In[92]:


balanced_target.shape


# 
# # Model Training
# 

# In[94]:


scaled_df_combined.groupby('loan_status').size()


# In[100]:


balanced_target_df = pd.DataFrame({"target":balanced_target})
balanced_target_df.groupby('target').size()


# In[101]:


from sklearn.model_selection import train_test_split


# In[106]:


x_train,x_test,y_train,y_test = train_test_split(balanced_features,balanced_target,test_size = 0.20,random_state=42)


# In[107]:


y_train.shape


# In[110]:


from sklearn.linear_model import LogisticRegression


# In[111]:


logit =  LogisticRegression()


# In[112]:


logit.fit(x_train,y_train)


# In[113]:


logit.score(x_train,y_train)


# In[114]:


logit_prediction = logit.predict(x_test)
logit_prediction


# In[116]:


from sklearn.metrics import classification_report


# In[117]:


logit_prediction


# In[118]:


print(classification_report(y_test,logit_prediction))


# In[122]:


print(logit.coef_[0])


# In[125]:


features_imp_logit = pd.DataFrame({"features":balanced_features.columns,"logit_impr":logit.coef_[0]})
features_imp_logit.head()                                                                                           


# In[128]:


from sklearn.ensemble import RandomForestClassifier


# # Random Forest Classifier
# 

# In[129]:


rf = RandomForestClassifier()


# In[130]:


rf.fit(x_train,y_train)


# In[132]:


rf.score(x_train,y_train)


# In[133]:


rf_prediction = rf.predict(x_test)
rf_prediction


# In[135]:


print(classification_report(y_test,rf_prediction))


# In[136]:


rf.feature_importances_[0]


# In[139]:


features_imp_rf = pd.DataFrame({"features":balanced_features.columns,"rf_impr":rf.feature_importances_[0]})
features_imp_rf.head()


# # XG BOOST

# In[146]:


pip install xgboost


# In[147]:


from xgboost import XGBClassifier


# In[149]:


xgb_model=  XGBClassifier(tree_method ='exact')


# In[150]:


xgb_model.fit(x_train,y_train.values.ravel())


# In[151]:


xgb_model.score(x_train,y_train.values.ravel())


# In[152]:


xgb_prediction = xgb_model.predict(x_test)
xgb_prediction 


# In[153]:


print(classification_report(y_test,xgb_prediction ))


# In[159]:


features_imp_xgb = pd.DataFrame({"features":balanced_features.columns, "xgb_imp":xgb_model.feature_importances_})
features_imp_xgb 


# In[160]:


feature_imp = pd.concat([features_imp_logit,features_imp_rf,features_imp_xgb],axis=1)


# In[161]:


feature_imp


# # Further Exploration

# In[163]:


xgb_prediction_df = pd.DataFrame({"test_indices":x_test.index, "xgb_pred": xgb_prediction})
xgb_prediction_df


# In[165]:


xgb_prediction_df = pd.DataFrame({"test_indices_xgb":x_test.index, "xgb_pred": xgb_prediction})
logit_prediction_df = pd.DataFrame({"test_indices_logit":x_test.index, "logit_pred": logit_prediction})
rf_prediction_df = pd.DataFrame({"test_indices_rf":x_test.index, "rf_pred": rf_prediction})


# In[169]:


merge_orig = cr_data.merge(xgb_prediction_df,left_index=True,right_on ='test_indices_xgb',how ='left' )


# In[170]:


merge_orig.head()


# In[171]:


merge_with_rf = merge_orig.merge(rf_prediction_df,left_index=True,right_on ='test_indices_rf',how ='left' )


# In[172]:


merge_with_rf


# In[173]:


merged_final =merge_with_rf.merge(logit_prediction_df,left_index=True,right_on ='test_indices_logit',how ='left' )


# In[174]:


merged_final


# In[176]:


merged_final.shape


# In[178]:


merged_final.dropna(inplace = True)


# In[179]:


merged_final.shape


# In[181]:


final_data_with_pred  = merged_final.drop(['test_indices_xgb','test_indices_logit','test_indices_rf'],axis=1)
final_data_with_pred .head()


# In[182]:


final_data_with_pred.to_excel(r"D:\Excel_Campus\pd_prediction.xlsx",index =False)


# In[ ]:




