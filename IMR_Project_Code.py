#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import datetime as dt
import re
from functools import partial
import itertools
from bioinfokit.analys import stat
import scipy.stats as stats
import seaborn as sns
import csv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from nltk.util import ngrams
nltk.download('all')


# In[2]:


### RYAN'S PORTION OF HW1 ###

# import csv into a pandas dataframe
imr_df = pd.read_csv(
    "C:\\Users\\Andrew\\Documents\\Syracuse University\\2023 Documents\\IST 652 Scripting for Data Analysis\\Final Project\\Independent_Medical_Review__IMR__Determinations__Trend.csv")


# In[3]:


# check dimensions
imr_df.shape


# In[4]:


# check data structure
imr_df.info()


# ### Data Munging

# In[5]:


# check for inf values
imr_df.isin([np.inf, -np.inf]).any()


# In[6]:


# check for nan values
imr_df.isna().sum().sort_values(ascending=False)


# In[7]:


# calculate number and perc of missing values by attribute
na_cnt = imr_df.agg(lambda x: (int(x.isna().sum()), '{:.2f}'.format((x.isna().sum() /
                        len(x))*100)))
na_cnt


# In[8]:


# use transpose to move row to header and header to row
na_df = na_cnt.transpose().sort_values(by=0, ascending=False)

# add column names
na_df.columns = ['Total NAs', 'Percent']
na_df


# In[9]:


# clean up findings column for data extraction
imr_df['findings_cleaned'] = imr_df['Findings'].str.replace('Nature of Statutory Criteria/Case Summary: ', '')
# convert column to strings and strip white space
imr_df['findings_cleaned'] = imr_df['findings_cleaned'].astype(str).str.strip()


# In[10]:


# avg word count per Findings
# create function to count words in string
def word_count(txt):
    words = str(txt).split()
    return len(words)

avg_word_count = imr_df['findings_cleaned'].apply(lambda x: word_count(x)).mean()
print('Average word count in Findings Column: ', round(avg_word_count))


# In[11]:


# find and extract ages in Findings. create new column with patients age
imr_df['Age'] = imr_df['findings_cleaned'].apply(lambda x: re.search(r'(\d+)-year', x).group(1) \
    if re.search(r'(\d+)-year', x) else None)


# In[12]:


# convert age to integer values
imr_df['Age'] = pd.to_numeric(imr_df['Age'], errors='coerce').astype('Int64')


# In[13]:


# determine correct age bin for new Age column
# determine bins
imr_df.groupby('Age Range').size()
bins = [0, 10, 20, 30, 40, 50, 65, 1000]
imr_df['age_bucket'] = pd.cut(imr_df['Age'],
                              bins=bins,
                              labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-64', '65+'])

# fill nan values from Age Range with values from age_bucket
imr_df['Age Range'].fillna(imr_df['age_bucket'], inplace=True)


# In[14]:


# replace 11_20 to 11-20 to stay consistent with data
imr_df['Age Range'] = imr_df['Age Range'].str.replace('_', '-')


# In[15]:


# find genders in Findings column
g_pattern = re.compile(r'\b(?:male|female)\b', flags=re.IGNORECASE)
imr_df['gender'] = imr_df['findings_cleaned'].apply(lambda x: re.search(g_pattern, x).group(0).title() \
    if re.search(g_pattern, x) else None)

# replace nan values in Patient Gender with gender column values
imr_df['Patient Gender'].fillna(imr_df['gender'], inplace=True)


# In[16]:


# review new nan values
na_cnt2 = imr_df.iloc[:, :-3].agg(lambda x: (int(x.isna().sum()), '{:.2f}'.format((x.isna().sum() / len(x)) * 100)))
na_df2 = na_cnt2.transpose().sort_values(by=0, ascending=False)
na_df2.columns = ['Total NAs', 'Percent']
na_df2


# In[17]:


# Bill Steel.  Date: 5/1/2023-5/6/2023
# Home Work 1

'''The following portion of HW1 examines 3 questions from the IMR data (Indepedent Medical Review) consisting 
of over 19000 medial reviews of a claim against an insurance plan's refusal of payment for services.  

The imr_df dataframe is a pandas dataframe based on .csv file obtained from Kaggle.

Question 1:  Are there any differences/biases in determination results based on gender?  
Question 2:  Are there any differences in outcomes based on Age Range? 
Question 3:  Are there any differences based on the type of Physician/Disease Condition?

Objectives:  Do analysis based on statistics and visually show results (e.g., bar charts)

A review of "Overturned" means the insurance company has been overruled, while "Upheld" 
means the IMR sided with the Insurance Company.'''


#baseline of total overturned vs. upheld
count=0
overturned=0
upheld=0
unknown=0
while count < len(imr_df):
    if imr_df.loc[count,'Determination']=='Overturned Decision of Health Plan':
        overturned=overturned+1
    else:
        if imr_df.loc[count,'Determination']=='Upheld Decision of Health Plan':
            upheld=upheld+1
        else:  #this statement checks for missing/corrupted data
            unknown=unknown+1
    count=count+1

print('overturned: {:,}, upheld: {:,}, unknown: {:,}'.format(overturned,upheld,unknown))
print('total records: {:,}'.format(len(imr_df)))
print('Percent overturned: {:.2f}%'.format(overturned/(overturned+upheld)*100))


# In[18]:


#Plot a bar chart of overall reviews showing upheld vs oveturned

data={'Overturned':overturned,'Upheld':upheld}
xaxis=list(data.keys())
yaxis=list(data.values())
fig=plt.figure(figsize=(3,3))
plt.bar(xaxis,yaxis,color='purple',width=.4)
plt.xlabel("Result of Findings",fontweight='bold',fontsize=8)
plt.ylabel("Number of Reviews",fontweight='bold',fontsize=8)
plt.title("Overturned vs. Upheld - Total",fontweight='bold',fontsize=8)
plt.show()


# In[19]:


#Determine overturned vs. upheld by gender

count=0
male=0
female=0
unknown=0
m_overturned=0
f_overturned=0
m_upheld=0
f_upheld=0
while count < len(imr_df):
    if imr_df.loc[count,'Patient Gender']=='Male':
        male=male+1
        if imr_df.loc[count,'Determination']=='Overturned Decision of Health Plan':
            m_overturned=m_overturned+1
        elif imr_df.loc[count,'Determination']=='Upheld Decision of Health Plan':
            m_upheld=m_upheld+1
    else:
        if imr_df.loc[count,'Patient Gender']=='Female':
            female=female+1
            if imr_df.loc[count,'Determination']=='Overturned Decision of Health Plan':
                f_overturned=f_overturned+1
            elif imr_df.loc[count,'Determination']=='Upheld Decision of Health Plan':
                f_upheld=f_upheld+1
        else:  #this statement checks for missing/corrupted data
            unknown=unknown+1
    count=count+1

percent_female = round(f_overturned/female*100,2)
percent_male = round(m_overturned/male*100,2)

print('Male: {}, Female: {}, missing/other: {}'.format(male,female,unknown))
print('Percent of claims related to Males {:.2f}%'.format(male/(male+female)*100))
print('Percent of claims related to Females {:.2f}%'.format(female/(male+female)*100))
print('total records: {}'.format(len(imr_df)))
print('Total valid records =',(male+female))
print('Male Overturned',m_overturned)
print('Male Upheld',m_upheld)
print('Female Overturned',f_overturned)
print('Female Upheld',f_upheld)
print('Female Overturned {:.2f}%'.format(percent_female))
print('Male Overturned {:.2f}%'.format(percent_male))


# In[20]:


#Builds the plot showing overturned vs. upheld by total, males, and females

barWidth=.25
fig=plt.subplots(figsize=(12,8))

total=[overturned,upheld]
male_total=[m_overturned,m_upheld]
female_total=[f_overturned,f_upheld]
      
br1=np.arange(len(total))
br2=[x+barWidth for x in br1]
br3=[x+barWidth for x in br2]

plt.bar(br1,total,color='purple', width=barWidth,edgecolor='grey',label='Total')
plt.bar(br2,male_total,color='b',width=barWidth,edgecolor='grey',label='Male')
plt.bar(br3,female_total,color='pink',width=barWidth,edgecolor='grey',label='Female')

plt.xlabel('Result of Findings',fontweight='bold',fontsize=15)
plt.ylabel('Number of Reviews', fontweight='bold',fontsize=15)
plt.title("Overturned vs. Upheld - by Gender",fontweight='bold',fontsize=15)
plt.xticks([r+barWidth for r in range(len(total))],['Overturned', 'Upheld'],fontsize=15)
plt.legend()


# In[21]:


#Overturned by age group
x=imr_df.groupby(['Determination','Age Range'])['Reference ID'].count()
imr_grouped_list=[]
row=[]
count=0
while count < len(x):
    row=[x.index[count][0],x.index[count][1],x[count]]
    imr_grouped_list.append(row)
    count += 1
df=pd.DataFrame(imr_grouped_list,columns=['Determination','Age Range','Total'])

o_percent_by_age_group=[]
count=0
while count<int(len(df)/2):
    percent=round((int(df.loc[count][2])/(int(df.loc[count][2])+int(df.loc[count+7][2])))*100,2)
    age_group=df.loc[count][1]
    o_percent_by_age_group.append([age_group,percent])
    count += 1

percent_df=pd.DataFrame(o_percent_by_age_group,columns=['Age Range','Percent Overturned'])
percent_df


# In[22]:


#Overturned by age goup plot
df.pivot(index="Age Range", 
         columns= "Determination", 
         values = "Total").plot(kind='bar')

plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.ylabel('Number of Reviews', fontweight='bold',fontsize=10)
plt.show()


# In[23]:


#Percent overturned by Age Group
percent_df=pd.DataFrame(o_percent_by_age_group,columns=['Age Range','Percent Overturned'])
percent_df.set_index('Age Range', inplace=True)
percent_df.plot(kind='bar')


# In[24]:


#Question:  Is there a relationship between the outcome of an IMR determination 
#and the medical specialty of the physician who made the determination? 

x=imr_df.groupby(['Determination','Diagnosis Category'])['Reference ID'].count()
imr_grouped_list2=[]
row=[]
count=0
while count < len(x):
    row=[x.index[count][0],x.index[count][1],x[count]]
    imr_grouped_list2.append(row)
    count += 1

imr_grouped_list2

df=pd.DataFrame(imr_grouped_list2,columns=['Determination','Physician Type','Total'])

o_percent_by_pt=[]
count=0
while count<int(len(df)/2):
    percent=round((int(df.loc[count][2])/(int(df.loc[count][2])+int(df.loc[count+29][2])))*100,2)
    total_reviews=df.loc[count][2]+df.loc[count+29][2]
    pt_group=df.loc[count][1]
    o_percent_by_pt.append([pt_group,percent,total_reviews])
    count += 1

percent_df=pd.DataFrame(o_percent_by_pt,columns=['Physician Type','Percent Overturned','Total Reviews'])
percent_df.sort_values('Total Reviews', ascending=False)


# In[25]:


#Prepare the plot of determination by Physician Type

df.pivot(index="Physician Type", 
         columns="Determination",
         values="Total").plot(kind='bar')

plt.rc('xtick',labelsize=5)

plt.ylabel('Number of Reviews', fontweight='bold',fontsize=10)
plt.xlabel('Physician Type', fontweight='bold',fontsize=10)

plt.show()


# In[26]:


#Percent overturned by Physician Type

percent_df=pd.DataFrame(o_percent_by_pt,columns=['Physician Type','Percent Overturned','Total Reviews'])
percent_df.set_index('Physician Type', inplace=True)
percent_df.drop('Total Reviews',axis=1,inplace=True)
percent_df.plot(kind='bar',fontsize=8)


# # Andrew's Part of the Project

# In[27]:


#Using string formatting to get a general overview of the dataset. 

print(f"The dataset has {len(imr_df.columns)} columns and {len(imr_df)} rows. \
It uses metrics from {len(imr_df['Report Year'].unique())} years, \
{len(imr_df['Diagnosis Category'].unique())} diagnosis categories, \
{len(imr_df['Diagnosis Sub Category'].unique())} diagnosis sub categories, \
{len(imr_df['Treatment Category'].unique())} treatment categories, \
{len(imr_df['Treatment Sub Category'].unique())} treatment sub categories, \
whether or not the determination was upheld or overturned, \
{len(imr_df['Type'].unique())} determination types, \
{len(imr_df['Age Range'].unique())} age ranges, \
{len(imr_df['Patient Gender'].unique())} patient genders, \
and a report on findings for each reference ID")


# In[28]:


#This is a bar chart that shows the distribution of cases by year. 

#Creating bar chart
year_counts = imr_df.groupby('Report Year').size().reset_index(name='Number of cases')
plt.bar(year_counts['Report Year'], year_counts['Number of cases'])
plt.xlabel('Year')
plt.ylabel('Number of cases')
plt.grid(zorder=0)
plt.xticks(year_counts['Report Year'], rotation=45)
plt.gca().set_axisbelow(True)
plt.show() 


# In[29]:


#This horizontal bar plot shows the distribution of diagnosis categories.

#Creating horizontal bar chart
diagnosis_counts = imr_df.groupby('Diagnosis Category')['Reference ID'].count()
colors = sns.color_palette('Set1', n_colors=len(diagnosis_counts))
plt.barh(diagnosis_counts.index, diagnosis_counts.values, color=colors)
plt.xlabel('Number of Cases')
plt.ylabel('Diagnosis Category')
plt.title('Diagnosis Category Distribution')
plt.grid(axis='x',zorder=0)
plt.yticks(np.arange(len(diagnosis_counts)), diagnosis_counts.index, fontsize=10, y=0.02)
plt.show() 


# In[30]:


#This is a vertical bar chart that shows the distribution of treatment categories.

#Creating bar chart
tc = imr_df.groupby('Treatment Category').size().reset_index(name='Number of cases')
ftc = tc[tc['Number of cases'] >= 1]
sftc = ftc.sort_values('Number of cases', ascending=False)
plt.bar(sftc['Treatment Category'], sftc['Number of cases'])
plt.title('Distribution of Treatment Categories')
plt.xlabel('Treatment Category')
plt.ylabel('Number of cases')
plt.grid(zorder=0)
plt.xticks(sftc['Treatment Category'], rotation=90)
plt.gca().set_axisbelow(True)
plt.show()


# In[31]:


#This is a chart that shows the percent of overturned or upheld determinations.

#Creating pie chart
counts = imr_df['Determination'].value_counts()
colors = ['#1f77b4', '#ff7f0e']
ax = counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, 
            startangle=90, counterclock=False, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
ax.set_ylabel('')
plt.axis('equal')
plt.title('Determination')
plt.show()


# In[ ]:





# In[32]:


#This chart shows the percentage distribution of case medical types.

#Creating pie chart
counts = imr_df['Type'].value_counts()
colors = ['#1f77b4', '#ff7f0e', '#9467bd']
ax = counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, 
            startangle=90, counterclock=False, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
ax.set_ylabel('')
plt.axis('equal')
plt.title('Type')
plt.show()


# In[ ]:





# In[33]:


#This chart shows the percentage distribution of age ranges.

#Creating pie chart
counts = imr_df['Age Range'].value_counts()
colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728', '#8c564b', '#bcbd22']
ax = counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, 
            startangle=90, counterclock=False, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
ax.set_ylabel('')
plt.axis('equal')
plt.title('Age Ranges')
plt.show()


# In[34]:


#This chart shows the percentage distribution of male and female patient sex

#Creating pie chart
counts = imr_df['Patient Gender'].value_counts()
colors = ['#e377c2', '#1f77b4']
ax = counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, 
            startangle=90, counterclock=False, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
ax.set_ylabel('')
plt.axis('equal')
plt.title('Patient Gender')
plt.show()


# In[35]:


#This bar chart shows the distribution of upheld and overturned determinations by year.

#Creating bar chart
counts = imr_df.groupby(['Report Year', 'Determination']).size().reset_index(name='counts')
pivot = counts.pivot(index='Report Year', columns='Determination', values='counts')
ax = pivot.plot(kind='bar', stacked=False)
ax.set_axisbelow(True)
ax.grid(zorder=0)
plt.title('Determination by Year')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.show()


# # Andrew's Deep Dive Analysis on Cancer

# In[36]:


imr_df.head()


# In[37]:


cancer_df = imr_df[imr_df['Diagnosis Category'] == 'Cancer']

cancer_df['Diagnosis Sub Category'].unique()


# In[38]:


# Group the DataFrame by age range and patient gender and count the occurrences
grouped_df = cancer_df.groupby(['Age Range', 'Patient Gender']).size().unstack()
grouped_df.plot(kind='bar', stacked=True, color=['pink', 'blue'])
plt.title('Cancer Diagnoses by Age Range and Patient Sex')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.show()


# In[39]:


# Count the occurrences of each cancer subtype
subtype_counts = cancer_df['Diagnosis Sub Category'].value_counts()
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
subtype_counts.plot(kind='bar')
plt.title('Distribution of Cancer Sub Categories')
plt.xlabel('Cancer Sub Categories')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[40]:


# Count the occurrences of each cancer subtype
subtype_counts = cancer_df['Treatment Category'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
subtype_counts.plot(kind='bar')
plt.title('Distribution of Cancer Treatment Categories')
plt.xlabel('Treatment Categories')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[41]:


# Count the occurrences of each cancer subtype
subtype_counts = cancer_df['Treatment Sub Category'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
subtype_counts.plot(kind='bar')
plt.title('Distribution of Cancer Treatment Sub Categories')
plt.xlabel('Treatment Sub Categories')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# # Andrew's Positive/Negative Sentiment Analysis

# In[42]:


#Duplicating dataset to not alter the original
andrew_imr = imr_df


# In[43]:


#Adding sentiment analysis and sentiment columns
from textblob import TextBlob

andrew_imr['sentiment'] = andrew_imr['findings_cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)

andrew_imr['sentiment_label'] = andrew_imr['sentiment'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

andrew_imr.head()


# In[44]:


# Count the frequency of sentiment labels
sentiment_counts = andrew_imr['sentiment_label'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment Labels')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Labels')
plt.show()


# In[45]:


# Create a histogram of sentiment scores
plt.hist(andrew_imr['sentiment'], bins=20, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.show()


# In[46]:


# Select the top 5 categories based on frequency
top_categories = andrew_imr['Diagnosis Category'].value_counts().nlargest(5).index
data = [andrew_imr[andrew_imr['Diagnosis Category'] == category]['sentiment'] for category in top_categories]
plt.boxplot(data)
plt.xticks(range(1, len(top_categories) + 1), top_categories, rotation=90)
plt.xlabel('Diagnosis Category')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores by Top 5 Diagnosis Categories')
plt.show()


# In[47]:


from wordcloud import WordCloud, STOPWORDS

# Define your additional stop words
additional_stopwords = {'patient', 'treatment', 'patients', 'treatments', 'year', 'old', 'medical', 'physician'}

positive_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Positive']['findings_cleaned'])
negative_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Negative']['findings_cleaned'])
neutral_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Neutral']['findings_cleaned'])

# Creating positive word cloud
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(additional_stopwords)).generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Sentiment')
plt.show()

# Creating negative word cloud
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(additional_stopwords)).generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Sentiment')
plt.show()

# Creating neutral word cloud
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(additional_stopwords)).generate(neutral_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Neutral Sentiment')
plt.show()


# In[ ]:





# In[48]:


from wordcloud import WordCloud, STOPWORDS

# Define your additional stop words
new_stopwords = {

    'medical', 'condition', 'treatment', 'enrollee', 'authorization', 'physician', 'reviewer',

    'coverage', 'therapy', 'necessary', 'medically', 'patient', 'procedure', 'health', 'plan',

    'likely', 'beneficial', 'available', 'reviewers', 'denied', 'request', 'proposed', 'experience',

    'expert', 'knowledgeable', 'recent', 'documentation', 'submitted', 'current', 'clinical',

    'result', 'actual', 'independent', 'review', 'denial', 'overturned', 'actively', 'practicing',

    'requested', 'found', 'determined', 'therefore', 'board', 'certified', 'et', 'upon', 'information',

    'experts', 'layperson', 'level', 'standard', 'similar','three', 'language', 'necessity.', 'demonstrate',

    'services', 'issue', 'american', 'performed', 'upheld', 'basis', 'records', 'certification', 'administration',

    'based', 'evidence', 'S', 'forth', 'necessity', 'week'

}

positive_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Positive']['findings_cleaned'])
negative_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Negative']['findings_cleaned'])
neutral_text = ' '.join(andrew_imr[andrew_imr['sentiment_label'] == 'Neutral']['findings_cleaned'])

# Creating positive word cloud
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(new_stopwords)).generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Sentiment')
plt.show()

# Creating negative word cloud
#wordcloud_negative = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(additional_stopwords)).generate(negative_text)
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud_negative, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud - Negative Sentiment')
#plt.show()

# Creating neutral word cloud
#wordcloud_neutral = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS.union(additional_stopwords)).generate(neutral_text)
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud_neutral, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud - Neutral Sentiment')
#plt.show()


# ### Natural Language Processing

# In[49]:


# remove stop words/punctuations
stop_words = set(stopwords.words('english'))
# add additional stop words
stpwrd = nltk.corpus.stopwords.words('english')
new_stopwords = ['medical', 'condition', 'treatment', 'enrollee', 'authorization', \
                 'coverage', 'therapy', 'necessary', 'medically', 'patient', 'procedure', \
                 'likely', 'beneficial', 'available']
stpwrd.extend(new_stopwords)


# In[50]:


# use regex to extract all words after the first occurrence of "requested" up to the end of the sentence.
#req_text = imr_df['findings_cleaned'].apply(lambda x: re.search(r'requested\s+([^\.]+)\.', x).group(1) \
#    if re.search(r'requested\s+([^\.]+)\.', x) else "")


# In[51]:


# removes all text before and including the word "Findings:"
req_text = imr_df['findings_cleaned'].apply(lambda x: re.search(r'Findings:\s*(.*)', x).group(1).lstrip() if re.search(r'Findings:\s*(.*)', x) else x)


# In[52]:


req_text[1]


# In[53]:


type(req_text)


# In[54]:


# initialize empty list
req_list = []
# loop through text and convert text to lowercase and remove stopwords
for index, row in req_text.items():
    sentences = sent_tokenize(row)
    cleaned_words = []
    for s in sentences:
        words = word_tokenize(s)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stpwrd]
        # ''.join() converts lists back into single strings
        cleaned_words.append(' '.join(words))
    req_list.append('. '.join(cleaned_words))


# In[55]:


# n-grams: collection of 3 successive items in req_list
all_ngrams = [ngrams(sentence.split(), 3) for sentence in req_list]
# flatten the list of n-grams
flattened_ngrams = [ngram for sublist in all_ngrams for ngram in sublist]
counter = Counter(flattened_ngrams)
# most common n-grams
most_common = counter.most_common(10)


# In[56]:


type(most_common)


# In[57]:


#  loop through list and print out most common pair of words
print("\nTop 10 most common trigrams in patient's findings:\n")
for i, ((w1, w2, w3), count) in enumerate(most_common, 1):
    print(f'{i}: {w1} {w2} {w3} \nOccurrences: {count}\n')


# ### Statistical Testing

# In[58]:


# create a contingency table fo Type and Determinaton
ct = pd.crosstab(index=imr_df['Type'],
                 columns=imr_df['Determination'])
ct


# In[59]:


# perform chi-square test of independence to find relationships between attribute of interest
res = stat()
res.chisq(df=ct)
print(res.summary)


# In[60]:


# Gender v Determination
gd = pd.crosstab(index=imr_df['Patient Gender'],
                 columns=imr_df['Determination'],
                 margins=True)
res.chisq(df=gd)
print(res.summary)


# In[61]:


gd


# In[62]:


# Age v Determination
ad = pd.crosstab(index=imr_df['Age Range'],
                 columns=imr_df['Determination'],
                 margins=True)
res.chisq(df=ad)
print(res.summary)


# In[63]:


# Diagnosis v Determination
dd = pd.crosstab(index=imr_df['Diagnosis Category'],
                 columns=imr_df['Determination'],
                 margins=True)
res.chisq(df=dd)
print(res.summary)


# In[64]:


plt.style.use('seaborn')
#plt.style.use('tableau-colorblind10')
ax = imr_df.groupby(['Type', 'Determination']).size().unstack().plot(kind='bar')
totals = []
for i in ax.patches:
    totals.append(i.get_height())   
total = sum(totals)

# add percentages to the bars
for i in ax.patches:
    ax.text(i.get_x(), i.get_height() + 50, '{:.2%}'.format(i.get_height() / total), fontsize=10)
# show the plot
plt.xlabel('Medical Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[65]:


# Diagnosis v Determination
dd = pd.crosstab(index=imr_df['Diagnosis Category'],
                 columns=imr_df['Determination'],
                 margins=True)
res.chisq(df=dd)
print(res.summary)

plt.style.use('seaborn')
#plt.style.use('tableau-colorblind10')
ax = imr_df.groupby(['Type', 'Determination']).size().unstack().plot(kind='bar')
totals = []
for i in ax.patches:
    totals.append(i.get_height())   
total = sum(totals)

# add percentages to the bars
for i in ax.patches:
    ax.text(i.get_x(), i.get_height() + 50, '{:.2%}'.format(i.get_height() / total), fontsize=10)
# show the plot
plt.xlabel('Medical Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

