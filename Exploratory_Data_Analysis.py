#!/usr/bin/env python
# coding: utf-8

# # Research on car sales ads
# 
# You will have the data from a real estate agency. It is an archive of sales ads for realty in St. Petersburg, Russia, and the surrounding areas collected over the past few years. You’ll need to learn how to determine the market value of real estate properties. Your task is to define the parameters. This will make it possible to build an automated system that is capable of detecting anomalies and fraudulent activity.
# 
# There are two different types of data available for every apartment for sale. The first type is a user’s input. The second type is received automatically based upon the map data. For example, the distance from the city center, airport, the nearest park or body of water. 

# ### Step 1. Open the data file and study the general information. 

# In[1]:


import pandas as pd
import numpy as np 
df = pd.read_csv('/datasets/vehicles_us.csv')
print('---Opened the data file and studied the general information---')
df.info()#getting imformation about datatypes and columns
df.describe()#getting information of the table mean, max values length etc. 
df


# ### Conclusion

# The dataset contains the following fields:
# •	price
# •	model_year
# •	model
# •	condition
# •	cylinders
# •	fuel — gas, diesel, etc.
# •	odometer — the vehicle's mileage when the ad was published
# •	transmission
# •	paint_color
# •	is_4wd — whether the vehicle has 4-wheel drive (Boolean type)
# •	date_posted — the date the ad was published
# •	days_listed — from publication to removal
# Result is printed with datatype information and general description.
# 
# In this we can analyse the price of second hand vehicle by considering all factors with wich we can predict how market is going currently. what things are in demand regarding this business? and how we can improve it.

# ### Step 2. Data preprocessing

# In[2]:


print(df.isnull().sum())#checked the missing values

df['model_year_noempty']= df.groupby('model')['model_year'].transform(lambda y : y.fillna(y.mode()[0]))#used mode method tofill values with most common year used for that particular model
df['odometer_noempty']= df.groupby('model_year_noempty')['odometer'].transform(lambda y : y.fillna(y.mean()))# used mean to fill mileage with all the vlues mean used in that perticular year
df['paint_color_noempty']= df['paint_color'].fillna('unknown')#filling paint color with unknown because we don't have info for it
df['cylinders_noempty']= df.groupby('model')['cylinders'].transform(lambda y : y.fillna(y.mode()[0]))# filling up missing values with most common number of cylinders used with perticular model


convert_dict = {'model_year_noempty': int, 
                'cylinders_noempty': int
               } 
  
df = df.astype(convert_dict) #changed datatypes into int for further calculation
df.info()
df


# 1. It looks like some customers forgot to add model year or may be they don't have much information on it so we changed it by grouping model year with model and took most common year given for it.
# 2. To Fill missing values in the Mileage we used group by method and filled it with mean so it should not affect our data and further analysis.
# 3. Paint color we filled with Unknown as we can't predict it but I think User has forgot to put it or may be just skipped this stage to fill it.
# 4. for missing cylinders we used same logic as of model year above.

# ### Step 3. Make calculations and add them to the table

# In[3]:


condition_dict = {'salvage':0,
                 'fair':1,
                 'good':2,
                 'excellent':3,
                 'like new':4,
                 'new':5}
df['condition_id']=df['condition'].apply(lambda x:condition_dict[x])#changing condition column with index values and savve it in new column
df['condition_id'].value_counts()
df['DateTime'] = df['date_posted'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))#used datetime format
df['weekday'] = df['DateTime'].dt.weekday#finding weekday
df['dayofmonth'] = df['DateTime'].dt.day#finding dayof the month
df['dayofyear'] = df['DateTime'].dt.dayofyear#finding day of the year
df['year'] = pd.DatetimeIndex(df['DateTime']).year#taking year out from date posted
df['age_of_the_vehicle'] = df['year'] - df['model_year_noempty'] #calculated age of the vehicle by posting year and model year
df.loc[df['age_of_the_vehicle'] == 0,'age_of_the_vehicle']= 1#changed 0 to 1 as we need this values in further calculation

df['avg_mileage'] = df['odometer_noempty']/df['age_of_the_vehicle']#calculated avg mileage 
df


# Calculated following parameters:
# •	Day of the week, month, and year the ad was placed
# •	The vehicle's age (in years) when the ad was placed
# •	The vehicle's average mileage per year
# This will give an idea about our first conclusion mentioned above to get an idea of what is going on in current market for this business.
# 

# ### Step 4. Carry out exploratory data analysis

# In[4]:


import pylab as pl
import matplotlib.pyplot as plot

columns_to_be_cleaned = ['age_of_the_vehicle','price', 'condition_id','odometer_noempty','cylinders_noempty','days_listed']
#Used for loop to plot graphs for given inforamation with and without outliers
for x in columns_to_be_cleaned:
    Q1_vehicle_age = df[x].quantile(0.25)
    print(Q1_vehicle_age)
    Q3_vehicle_age = df[x].quantile(0.75)
    print(Q3_vehicle_age)
    IQR_vehicle_age = Q3_vehicle_age - Q1_vehicle_age
    print(IQR_vehicle_age)
    new_df = df.query(f"{x} >= @Q1_vehicle_age - 1.5*@IQR_vehicle_age and {x} <= @Q3_vehicle_age+1.5*@IQR_vehicle_age")
    df.hist(x,sharex=True, sharey=True)
    pl.title('Data With Outliers')
    pl.xlabel(f'{x}')
    pl.ylabel('count')
    new_df.hist(x,sharex=True, sharey=True)
    pl.title('Data Without Outliers')
    pl.xlabel(f'{x}')
    pl.ylabel('count')


# Here we studied different parameters of data by using histograms but also we eliminated unwanted eliments from data by using outliers to understand stability of the histograms.

# In[5]:


#print(new_df.sort_values('days_listed').head())
print('Mean of ads displyed for all given data:', new_df['days_listed'].mean())#calculated mean of raw data
print('Median of ads displyed for all given data:',new_df['days_listed'].median())#calculated median of raw data
print('Mode of ads displyed for all given data:',new_df['days_listed'].mode())#calculated mode to find out typical lifetime of ads 
new_df['long time'] = new_df['days_listed'] >= 104
print(new_df['long time'].value_counts())
new_df['short time'] = new_df['days_listed'] <= 0
print(new_df['short time'].value_counts())


# In[6]:


filtered_data = new_df.query('0 < days_listed < 104')
print('Mean of ads displyed for filtered data:',filtered_data['days_listed'].mean())
print('Mean of ads displyed for filtered data:',filtered_data['days_listed'].median())
print('Mean of ads displyed for filtered data:',filtered_data['days_listed'].mode())#calculated mode to find out typical lifetime of ads
filtered_data.hist('days_listed',sharex=True, sharey=True)
pl.title('Number Of days Ads were displayed with filter data')
pl.xlabel('days_listed')
pl.ylabel('Count')


# Here we studied the advertisement lifespan. we stored data when ads were removed quickly, and when they were listed for an abnormally long time. and also calculated mode to find out typical lifetime of advertisement.
# quick removal is o days which are 54 ads removed same day.
# long time is 104 days which are 83 ads removed after long days.

# In[5]:


ad_pivot = new_df.pivot_table(index = 'type', values = 'price',aggfunc = 'mean')
print(ad_pivot.sort_values('price'))#calculated average price for each type of vehicle
import matplotlib.pyplot as plt
ad_pivot_1 = new_df.pivot_table(index = 'type', values = 'price',aggfunc = 'count')
ad_pivot1_sorted = ad_pivot_1.sort_values('price',ascending = False)
print(ad_pivot1_sorted)
ad_pivot1_sorted.plot(sharex=True, sharey=True,kind='bar')#graph to show dependence of the number of ads on the vehicle type
pl.title('Number Of days Ads for each type')
pl.xlabel('type')
pl.ylabel('number of ads')
print('---SUV and truck are the two types with the greatest number of ads---')


# Here we studied number of ads and the average price for each type of vehicle. The two types with the greatest number of ads are truck and SUV.

# In[36]:


filter_data_truck_rev = new_df.query('type == "truck"')
filter_data_truck_rev['paint_color'].value_counts()


# In[38]:



filter_data_truck = new_df.query('type == "truck" and paint_color_noempty != ("orange","purple")')#filtered data with more than 50 numbers of ads as per given condition
filter_data_pivot_truck = filter_data_truck.pivot_table(index = 'type', values = ['age_of_the_vehicle', 'odometer_noempty','condition_id','price'])
pd.plotting.scatter_matrix(filter_data_pivot_truck, figsize = (9, 9),grid=True)#Studying scatterplot matrix to find out price dependancy on the given parameters on each of the type we selected
filter_data_truck


# Here we studied to find out whether the price depends on age, mileage, condition, transmission type, and color. by analysing scattermatrix we can conclude that For Good Condition truck type model price is leading upto 17000 with old models upto seven and half year and which ran for less than 115000 kms.
# Yes we can say price is dependant on mostly condition and age of the vehicle.

# In[37]:


filter_data_suv = new_df.query('type == "SUV" and paint_color_noempty != ("orange","purple")')
filter_data_pivot_suv = filter_data_suv.pivot_table(index = 'type', values = ['age_of_the_vehicle', 'odometer_noempty','condition_id','price'])
pd.plotting.scatter_matrix(filter_data_pivot_suv, figsize = (9, 9),grid=True)#repeating steps from above for SUV type
filter_data_suv



# Here we studied to find out whether the price depends on age, mileage, condition, transmission type, and color. by analysing scattermatrix we can conclude that For Good to excellent Condition SUV type model price is leading upto 17000 with old models upto nine and half year and which ran for less than 130000 kms.
# Yes we can say price is dependant on mostly condition(Between good and Excellent) and age of the vehicle.

# In[43]:


filter_data_truck.boxplot(column=['price'], by='paint_color_noempty',
                     return_type='axes')
Q1 = filter_data_truck['price'].quantile(0.25)
Q3 = filter_data_truck['price'].quantile(0.75)
IQR = Q3 - Q1
plt.xticks(rotation=90)
plt.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red')#plotting boxplot for categorical variable colors with whiskers for truck



# Upon visual inspection of boxplots of price affected mostly by black color for truck  we can see that the price for color black is high range for truck
# 
# Drag have the lowest price range for Yellow and Brown color.

# In[40]:


filter_data_truck.boxplot(column=['price'], by='transmission',
                     return_type='axes')

plt.xticks(rotation=90)
plt.hlines(y= Q1, xmin=0.9, xmax=1.1, color='red')#plotting box plot for transmission type in truck


# Upon visual inspection of boxplots of price affected mostly by automatic transmission for truck  we can see that the price for automatic transmission is high range for truck
# 
# Drag have the lowest price range for manual transmission in truck.

# In[42]:


filter_data_suv.boxplot(column=['price'], by='paint_color_noempty',
                     return_type='axes')
Q1 = filter_data_suv['price'].quantile(0.25)
Q3 = filter_data_suv['price'].quantile(0.75)
IQR = Q3 - Q1
plt.xticks(rotation=90)
plt.hlines(y= [Q1-1.5*IQR], xmin=0.9, xmax=1.1, color='red') #repeating same for SUV



# Upon visual inspection of boxplots of price affected mostly by black and red color for SUV  we can see that the price for these colors is high range for SUV
# 
# Drag have the lowest price range for  Brown and custom color.

# In[44]:


filter_data_suv.boxplot(column=['price'], by='transmission',
                     return_type='axes')
plt.xticks(rotation=90)
plt.hlines(y= Q1, xmin=0.9, xmax=1.1, color='red') #repeating same for SUV



# Upon visual inspection of boxplots of price affected mostly manual transmission for SUV  we can see that the price for manual transmission is high range for SUV
# 
# Drag have the lowest price range for automatic transmission.

# ### Step 5. Overall conclusion

# Overall in conclusion I would like to conclude that factors influence the price of a vehicles are mostly Age, Vehicle type and Condition of the vehicle as we took two most popular type SUV and truck. The prorities of customer varied for other factors but the condition of the vehicle and age of the vehicle really got stable.  Condition parameter was equal to or more than Good and age varied upto 9 years.

# ### Project completion checklist
# 
# Mark the completed tasks with 'x'. Then press Shift+Enter.

# - [x]  file opened
# - [x]  files explored (first rows printed, info() method)
# - [x]  missing values determined
# - [x]  missing values filled in
# - [x]  clarification of the discovered missing values provided
# - [x]  data types converted
# - [x]  explanation of which columns had the data types changed and why
# - [x]  calculated and added to the table: day of the week, month, and year the ad was placed
# - [x]  calculated and added to the table: the vehicle's age (in years) when the ad was placed
# - [x]  calculated and added to the table: the vehicle's average mileage per year
# - [x]  the following parameters investigated: price, vehicle's age when the ad was placed, mileage, number of cylinders, and condition
# - [x]  histograms for each parameter created
# - [x]  task completed: "Determine the upper limits of outliers, remove the outliers and store them in a separate DataFrame, and continue your work with the filtered data."
# - [x]  task completed: "Use the filtered data to plot new histograms. Compare them with the earlier histograms (the ones that included outliers). Draw conclusions for each histogram."
# - [x]  task completed: "Study how many days advertisements were displayed (days_listed). Plot a histogram. Calculate the mean and median. Describe the typical lifetime of an ad. Determine when ads were removed quickly, and when they were listed for an abnormally long time.  "
# - [x]  task completed: "Analyze the number of ads and the average price for each type of vehicle. Plot a graph showing the dependence of the number of ads on the vehicle type. Select the two types with the greatest number of ads. "
# - [x]  task completed: "What factors impact the price most? Take each of the popular types you detected at the previous stage and study whether the price depends on age, mileage, condition, transmission type, and color. For categorical variables (transmission type and color), plot box-and-whisker charts, and create scatterplots for the rest. When analyzing categorical variables, note that the categories must have at least 50 ads; otherwise, their parameters won't be valid for analysis.  "
# - [x]  each stage has a conclusion
# - [x]  overall conclusion drawn
