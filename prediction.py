#!/usr/bin/env python
# coding: utf-8

# ### Analyzing the Available Data

# In[1]:


#Importing required libraries

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import random
from IPython.display import Image
random_num=122


# In[6]:


#Reading the dataset

df=pd.read_csv("C:\\Users\\John\\Downloads\\international_matches (2).csv")
df.head(10)


# In[ ]:


df.describe()


# In[7]:


#Preprocessing
#Finding outmissing values

df.isnull().sum()
#No any missing values
#We are good to go


# In[10]:


#finding outliers in the numerical data columns

fig = plt.figure(figsize =(10, 5))
plt.boxplot([df["home_team_score"],df["away_team_score"]])
plt.xticks([1,2], ["Home Score", "Away Score"])


# In[11]:


#Boxplot says there are many outliers
#Let's remove those outlier,It will make our Machine Learning model more accurate
#Asume maximum goals that one team can score is 15

higher_home=15
higher_away=15
df=df[(df["home_team_score"]<higher_home) & (df["away_team_score"]<higher_away)]
df.head(5)
#Dataset after removing outliers


# In[12]:


#Making a new colum named "Win_Statues" to store the result(Win,Lost,Draw) of the home team

conditions  = [df["home_team_score"] ==df["away_team_score"], df["home_team_score"] > df["away_team_score"] , df["home_team_score"] < df["away_team_score"]]
choices     = [ "Draw", 'Win', 'Lost' ]
df["Win_Statues"] = np.select(conditions, choices)
df.head(5)


# In[13]:


#See what are the Countries in the dataset

countries=df.home_team.unique()
print(f"There are {len(countries)} Countries in the home_team Column\n")
print(f"Countries-{countries}")


# **Type of the Matches**

# In[14]:


rank_bound = 10
ax = df.tournament.value_counts()[:rank_bound].sort_values()
value = ax.values
label = ax.index

plt.figure(figsize=(14,6))
plt.barh(y=label, width=value, edgecolor="k")
for i in range(rank_bound):
    plt.text(x=50,y=i-0.1,s=value[i],color="w",fontsize=12)
plt.show()


# **Teams played most Matches**

# In[15]:


rank_bound = 10
ax = df.country.value_counts()[:rank_bound].sort_values()
value = ax.values
label = ax.index

plt.figure(figsize=(14,6))
plt.barh(y=label, width=value, edgecolor="k")
for i in range(rank_bound):
    plt.text(x=10,y=i-0.1,s=value[i],color="w",fontsize=12)
plt.show()


# **Number of Matches played in equal intervals**

# In[16]:


years = []
for date in df.date:
    years.append(int(str(date)[0:4]))
plt.figure(figsize=(14,6))
plt.hist(years, density=True, bins=12, edgecolor="k")
plt.title("Histogram of Years")
plt.ylabel("Frequency")
plt.xlabel("Year")
plt.show()

#Most matches are played after 1960


# ### (1) Finding out the impact hosting a major tournament helps a country's chances in winning matches?

# In[17]:


#Now take the rpws that home_team==Country 
#Nad romove the data of friendly matches

data_home=df.loc[df["home_team"]==df["country"] ]
data_home=df.loc[df["tournament"] != "Friendly"]
data_home.head(5)


# **Home Team Winning Statistcs**

# In[18]:


#How home team performs in tournament matches

tournament_df=pd.crosstab(data_home["tournament"], data_home["Win_Statues"],margins=True)
tournament_df=tournament_df.sort_values("All",ascending=False).head(10)
tournament_df.style.bar(color="brown",subset=["Draw","Lost","Win","All"])


# In[19]:


#Home team results
sns.displot(data_home, x="Win_Statues")
plt.title("Home Team Winning Status")


# In[20]:


data_home['Win_Statues'].value_counts()


# In[21]:


#How each team perform while playing a tournament in home grounds

teams_win_statues=pd.crosstab(df["home_team"], df["Win_Statues"],margins=True, margins_name="Total")
teams_win_statues["team_win_probability"]=teams_win_statues["Win"]/(teams_win_statues["Total"])
#Lets take teams which plays atleast 200 games
teams_win_statues_100=teams_win_statues.loc[teams_win_statues["Total"]>200]
teams_win_statues_100=teams_win_statues_100.sort_values("team_win_probability",ascending=False)
teams_win_statues_100.head(20).style.bar(color="orange",subset="team_win_probability")

#We can see teams has more than 50% wining probability while playing in the home grounds 


# **Away team winning statistics**

# In[22]:


#Teams playing in away conditions

teams_away_statues=pd.crosstab(df["away_team"], df["Win_Statues"],margins=True, margins_name="Total")
teams_away_statues["team_win_probability"]=teams_away_statues["Lost"]/(teams_away_statues["Total"])
#Lets take teams which plays atleast 200 games
teams_away_statues_100=teams_away_statues.loc[teams_away_statues["Total"]>200]
teams_away_statues_100=teams_away_statues_100.sort_values("team_win_probability",ascending=False)
teams_away_statues_100.rename(columns={'Lost': 'Win'}, index={'Win': 'Lost'}, inplace=True)
teams_away_statues_100.head(20)

#The probability to wining a tournament in away conditions is very low comapred to the winning probalities in home conditions


# In[23]:


#Making a plot to identify wining possibilities in both home and away conditions

win_prob={teams_win_statues_100.iloc[i,:].name:[teams_win_statues_100.iloc[i,4]] for i in range(teams_win_statues_100.shape[0])}
for i in range(teams_away_statues_100.shape[0]):
    try:
        win_prob[teams_away_statues_100.iloc[i,:].name].append(teams_away_statues_100.iloc[i,4]) 
    except:
        pass
country_names=win_prob.keys()
lst_home=[i[0] for i in win_prob.values()]

lst_away=[]
for i in win_prob.values():
    if len(i)==2:
        lst_away.append(i[1])
    else:
        lst_away.append(0)

plt.figure(figsize=(14,6))
plt.plot(country_names,lst_home, label = "Home Win Probability")
plt.plot(country_names, lst_away, label = "Away Win Probability")
plt.xticks(rotation = 90)
plt.title("Winning Probability of each team (Home and Away)")
plt.legend()
plt.show()


# Blue line is alway above the yellow line,showing home winning possibilies are always higher.
# 
# **Finally by analysing the data, we can identify that the home team has a clear edge while playing a tournament in their home grounds**
# 
# 

# ### (2) Finding Most Succesful Team
# 
# **Finding the team which performs best in home conditions each year interval**

# In[24]:


#Home Conditions
#Convert the data set into 10 terms and findsout who has done best at each term 

range_years=max(years)-min(years)
no_0f_terms=10
term_size=int(range_years/no_0f_terms)
for i in range(no_0f_terms+1):
    start=years.index(term_size*i+min(years))
    end=years.index(min(term_size*(i+1)+min(years),2021))
    term=df.iloc[start:end]
    best_teams=pd.crosstab(term["home_team"], term["Win_Statues"],margins=True, margins_name="Total")
    
    ##Lets take teams which plays atleast 20 games
    best_teams["team_win_probability"]=best_teams["Win"]/(best_teams["Total"])
    best_teams=best_teams.sort_values("team_win_probability",ascending=False)
    best_teams=best_teams.loc[best_teams["Total"]>20]
    if (best_teams.shape[0]>2):
        print(f"\nBest 2 team in the term: {term_size*i+min(years)}-{min(term_size*(i+1)+min(years),2021)} ")
        print(best_teams.iloc[0:2].to_markdown())
    else:
        print(f"No Enough data to find the best team in the term: {term_size*i+min(years)}-{min(term_size*(i+1)+min(years),2021)} ")


# **Finding the team which performs best in away conditions each year interval**

# In[25]:


#Away Conditions
##Convert the data set into 10 terms and findsout who has done best at each term 

range_years=max(years)-min(years)
no_0f_terms=10
term_size=int(range_years/no_0f_terms)
for i in range(no_0f_terms+1):
    start=years.index(term_size*i+min(years))
    end=years.index(min(term_size*(i+1)+min(years),2021))
    term=df.iloc[start:end]
    best_teams=pd.crosstab(term["away_team"], term["Win_Statues"],margins=True, margins_name="Total")
    
    ##Lets take teams which plays atleast 200 games
    best_teams["team_win_probability"]=best_teams["Lost"]/(best_teams["Total"])
    best_teams=best_teams.sort_values("team_win_probability",ascending=False)
    best_teams=best_teams.loc[best_teams["Total"]>20]
    print(f"\nBest 2 team in the term: {term_size*i+min(years)}-{min(term_size*(i+1)+min(years),2021)} ")
    if (best_teams.shape[0]>2):
        print(best_teams.iloc[0:2].to_markdown())
    else:
        print(f"No Enough data to find the best team in the term: {term_size*i+min(years)}-{min(term_size*(i+1)+min(years),2021)} ")


# ### (3) FIFA World Cup -2022- QATAR
# 
# **Making a ML model**

# In[26]:


df_match=df.copy() #making a copy of the main dataframe for future use
df_match.head(8)


# In[27]:


#Making a new dataset with required features to train the machine learning model
#Year,Played Country,Team_1,team_2,team_1 score,team_2 score

New_Dataset_part_1=pd.DataFrame(list(zip(years,df_match.values[:,7],df_match.values[:,1],df_match.values[:,2],df_match.values[:,3],df_match.values[:,4])),columns=["year","Country","team_1","team_2","team_1_score","team_2_score"])
#Making a new dataset by changing the team_1 and team_2 and their respective scores
New_Dataset_part_2=pd.DataFrame(list(zip(years,df_match.values[:,7],df_match.values[:,2],df_match.values[:,1],df_match.values[:,4],df_match.values[:,3])),columns=["year","Country","team_1","team_2","team_1_score","team_2_score"])
New_Dataset=pd.concat([New_Dataset_part_1,New_Dataset_part_2],axis=0)
New_Dataset =New_Dataset.sample(frac=1).reset_index(drop=True) #Shaffling the dataset
New_Dataset.head(5)


# In[44]:


#Creating a list containg all the names of the countries

# teams_1=New_Dataset.team_1.unique()
# contries=New_Dataset.Country.unique()
# all_countries=np.unique(np.concatenate((teams_1,contries), axis=0))
# len(all_countries)


import numpy as np

# Creating a list containing all the names of the countries
teams_1 = New_Dataset.team_1.unique()
countries = New_Dataset.Country.unique()

# Convert elements to strings and concatenate
all_countries = np.unique(np.concatenate((teams_1.astype(str), countries.astype(str)), axis=0))
len(all_countries)


# In[29]:


#Making a heatmap to see the correlation of each columns 

sns.heatmap(New_Dataset.corr())
New_Dataset.corr()


# In[45]:


#Defining the features and labels(Targets)

Y= New_Dataset.iloc[:,4:6] #Training targets (team_1_score and team_2_score)
categorized_data=New_Dataset.iloc[:,0:4].copy() #Traing features

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

#Labeling the data using LabelEncorder in Sklearn-(Giving a unique number to each string(country))

label_encoder.fit(all_countries)
#list(label_encoder.classes_)
categorized_data['team_1']=label_encoder.transform(categorized_data['team_1'])
categorized_data['team_2']=label_encoder.transform(categorized_data['team_2'])
categorized_data['Country']=label_encoder.transform(categorized_data['Country'])

#Converting these feature columns to categrize form to make the training processs more smoother
categorized_data['team_1']=categorized_data['team_1'].astype("category")
categorized_data['team_2']=categorized_data['team_2'].astype("category")
categorized_data['Country']=categorized_data['team_2'].astype("category")


# In[31]:


#Input Fatures to the model (x)

categorized_data.head(5)


# In[32]:


#Targets to the model (Y)

Y.head(5)


# In[33]:


#Info about the X and Y dataframes

print(categorized_data.info())
print(Y.info())


# In[63]:


#Making the model

X=categorized_data
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Use any algorithm
model = MultiOutputRegressor(RandomForestClassifier())
model.fit(X,Y)


# In[70]:


#Making the predictions

prd=model.predict(X)
prd


# In[78]:


#Creating the Confusion matrix for each predictions

score_team_1=[i[0] for i in prd]
score_team_2=[i[1] for i in prd]

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(list(Y.iloc[:,0]),score_team_1)
cm2=confusion_matrix(list(Y.iloc[:,1]),score_team_2)


# In[42]:


#Plotting the Confussion Matrix for score of team 01

plt.figure(figsize=(20,20))
sns.heatmap(cm1, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for Team 1 Score")
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[73]:


#Classification Report to team 1 Score

from sklearn.metrics import classification_report
report_1=classification_report(Y.iloc[:,0],score_team_1)
print(report_1)

#Has a good Accuracy in predicting the team 1 Score


# In[74]:


#Plotting the Confussion Matrix for score of team 02

plt.figure(figsize=(20,20))
sns.heatmap(cm2, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for team 2 score")
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[75]:


#Classification Report to team 2 Score

report_2=classification_report(Y.iloc[:,1],score_team_2)
print(report_2)

#Has a good Accuracy in predicting the team 1 Score#


# In[92]:


#Fuction to Select the winning team for the prediction array

def select_winning_team(probability_array):
    prob_lst=[round(probability_array[0][i],3) for i in range(2)]
    if (prob_lst[0]>prob_lst[1]):
        out=0
    elif (prob_lst[0]<prob_lst[1]):
        out=1
    elif (prob_lst[0]==prob_lst[1]):
        out=2
    return out,prob_lst


# In[101]:


#Sample Prediction

#

from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame named New_Dataset with training data
label_encoder_team = LabelEncoder()
label_encoder_country = LabelEncoder()

# Fit the label encoder on the relevant columns
encoded_team_1 = label_encoder_team.fit_transform(New_Dataset['team_1'])
encoded_country = label_encoder_country.fit_transform(New_Dataset['Country'])

# Manually encode the stadium name as a numerical value
# You can also use other encoding techniques for non-categorical data
stadium = "Qatar"
stadium_num = 1  # Assign a numerical value to Qatar

print(f"Team 01 is {team_1} - {team_1_num}")
print(f"Team 02 is {team_2} - {team_2_num}")
print(f"Played in {stadium} - {stadium_num}")


# In[113]:


#Sample Prediction Output

X_feature = np.array([[mactch_played, stadium_num, team_1_num, team_2_num]])

# Make predictions
res = model.predict(X_feature)

# Check the result and print the outcome
for prediction in res:
    if prediction[0] == 0:
        print(f"{team_1} vs {team_2}\n{team_1} wins 🏆⚽🎯\n")
    elif prediction[0] == 1:
        print(f"{team_1} vs {team_2}\n{team_2} wins 🏆⚽🎯\n")
    else:
        print(f"{team_1} vs {team_2}\nMatch Draw ⚽⚽⚽\n")



# **FIFA WORLD CUP - 2022 -**
# 
# I have selected 32 teams that has the maximum posibility to participate in FIFA World Cup 2022.
# These teams are selected accordingto the current world ranking and recently helg Euro Cup performances.
# 
# Every team plays with evey one-else(League System).That means one team plays 63 matches.Then from the model I predict who is going to win the match.
# * Winning team gets 02 Point
# * Match draw (Both teams scored same number of goals)- both teams get 1 points each
# * Team Lost gets no points
# 
# According this points system final winner will be selected

# <img src= "https://www.sinceindependence.com/wp-content/uploads/2019/12/fifa-world-cup-2022.jpg" alt ="Fifa">

# In[114]:


Group_A= ["Uruguay","Russia","Saudi Arabia","Egypt"]
Group_B= ["Portugal","Spain","Morocco","Iran"]
Group_C= ["France","Denmark","Australia","Peru"]
Group_D= ["Argentina","Croatia","Iceland","Nigeria"]
Group_E= ["Brazil","Switzerland","Costa Rica","Serbia"]
Group_F= ["South Korea","Mexico","Sweden","Germany"]
Group_G= ["Belgium","England","Panama","Tunisia"]
Group_H= ["Senegal","Japan","Poland","Colombia"]
Groups={"Group A":Group_A,"Group B":Group_B,"Group C":Group_C,"Group D":Group_D,"Group E":Group_E,"Group F":Group_F,"Group G":Group_G,"Group H":Group_H}


# In[118]:


#Initialize basic parameters

# year=2022
# stadium="Qatar"
# stadium_num=label_encoder.transform([stadium])[0]
# host_num=stadium_num
year = 2022
stadium = "Qatar"
stadium_num = 1  # Assign a numerical value to Qatar (or another value as appropriate)
host_num = stadium_num


# In[135]:


# ##Group stage Matches

# Group_standings={}
# for grp_name in list(Groups.keys()):
#     print(f"{grp_name} Matches")
#     probable_countries=Groups[grp_name]
#     team_wins_dct={}
#     goal_scored_dct={}
#     goal_against_dct={}
#     win_dct={}
#     draw_dct={}
#     lost_dct={}
#     for i in range(len(probable_countries)):
#         j=i+1
#         team_1=probable_countries[i]
#         team_1_num=label_encoder.transform([team_1])[0]
#         team_wins=0
#         while j<len((probable_countries)):
#             team_2=probable_countries[j]
#             team_2_num=label_encoder.transform([team_2])[0]
#             team_lst=[team_1,team_2]
#             Input_vector=np.array([[year,stadium_num,team_1_num,team_2_num]])
#             res=model.predict(Input_vector)

#             win,prob_lst=select_winning_team(res)
#             goal_scored_dct[team_1] = goal_scored_dct.get(team_1,0)+prob_lst[0]
#             goal_scored_dct[team_2] = goal_scored_dct.get(team_2,0)+prob_lst[1]

#             goal_against_dct[team_1] = goal_against_dct.get(team_1,0)+prob_lst[1]
#             goal_against_dct[team_2] = goal_against_dct.get(team_2,0)+prob_lst[0]

#             try:
#                 print(f" {team_1} vs {team_2} \n  Results of the Match {res[0]}\n   {team_lst[win]} wins 🎊🏆⚽🎖️\n")
#                 if (win)==0:
#                     team_wins_dct[team_1] = team_wins_dct.get(team_1,0)+2
#                     team_wins_dct[team_2] = team_wins_dct.get(team_2,0)
                    
#                     win_dct[team_1] = win_dct.get(team_1,0)+1
#                     win_dct[team_2] = win_dct.get(team_2,0)
#                     lost_dct[team_2] = lost_dct.get(team_2,0)+1
#                     lost_dct[team_1] = lost_dct.get(team_1,0)
#                     draw_dct[team_2] = draw_dct.get(team_2,0)
#                     draw_dct[team_1] = draw_dct.get(team_1,0)

#                 elif (win)==1:
#                     team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+2
#                     team_wins_dct[team_1] = team_wins_dct.get(team_1,0)
                    
#                     win_dct[team_2] = win_dct.get(team_2,0)+1
#                     win_dct[team_1] = win_dct.get(team_1,0)
#                     lost_dct[team_1] = lost_dct.get(team_1,0)+1
#                     lost_dct[team_2] = lost_dct.get(team_2,0)
#                     draw_dct[team_1] = draw_dct.get(team_1,0)
#                     draw_dct[team_2] = draw_dct.get(team_2,0)

#             except IndexError:
#                 print(f"{team_1} vs {team_2} \n  Results of the Match {res[0]}\n   Match Draw ⚽⚽⚽\n") 
#                 team_wins_dct[team_1] = team_wins_dct.get(team_1,0)+1
#                 team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+1
                
#                 draw_dct[team_1] = draw_dct.get(team_1,0)+1
#                 draw_dct[team_2] = draw_dct.get(team_2,0)+1
                
#                 win_dct[team_1] = win_dct.get(team_1,0)
#                 lost_dct[team_1] = lost_dct.get(team_1,0)
                
#                 win_dct[team_2] = win_dct.get(team_2,0)
#                 lost_dct[team_2] = lost_dct.get(team_2,0)
                    
#             j=j+1
#     group_results=[win_dct,draw_dct,lost_dct,team_wins_dct,goal_scored_dct,goal_against_dct]
#     Group_standings[grp_name]=group_results

# Assuming you have a label encoder for stadiums, similar to label_encoder_team
label_encoder_stadium = LabelEncoder()
label_encoder_stadium.fit(stadiums)

Group_standings = {}

# Iterate through the groups
for grp_name, probable_countries in Groups.items():
    print(f"{grp_name} Matches")
    
    team_wins_dct = {}
    goal_scored_dct = {}
    goal_against_dct = {}
    win_dct = {}
    draw_dct = {}
    lost_dct = {}
    
    # Iterate through the teams in the group
    for i in range(len(probable_countries)):
        for j in range(i + 1, len(probable_countries)):
            team_1 = probable_countries[i]
            team_2 = probable_countries[j]
            
            # Encode team and stadium
            team_1_num = label_encoder_team.transform([team_1])[0]
            team_2_num = label_encoder_team.transform([team_2])[0]
            stadium_num = label_encoder_stadium.transform([stadium])[0]
            
            # Create the feature vector for prediction
            X_feature = np.array([[year, stadium_num, team_1_num, team_2_num]])
            
            # Predict the match result
            res = model.predict(X_feature)
            
            # Get the winning team index (0 for team_1, 1 for team_2, -1 for draw)
            winning_team_index = select_winning_team(res)
            
            # Update statistics dictionaries
            goal_scored_dct[team_1] = goal_scored_dct.get(team_1, 0) + float(res[0][0])
            goal_scored_dct[team_2] = goal_scored_dct.get(team_2, 0) + float(res[0][1])
            
            goal_against_dct[team_1] = goal_against_dct.get(team_1, 0) + float(res[0][1])
            goal_against_dct[team_2] = goal_against_dct.get(team_2, 0) + float(res[0][0])
            
            try:
                print(f"{team_1} vs {team_2}\nResults of the Match: {res[0]}\n{prob_lst[winning_team_index]} wins 🎊🏆⚽🎖️\n")
                
                if winning_team_index == 0:
                    team_wins_dct[team_1] = team_wins_dct.get(team_1, 0) + 2
                    team_wins_dct[team_2] = team_wins_dct.get(team_2, 0)
                    win_dct[team_1] = win_dct.get(team_1, 0) + 1
                    win_dct[team_2] = win_dct.get(team_2, 0)
                    lost_dct[team_2] = lost_dct.get(team_2, 0) + 1
                    lost_dct[team_1] = lost_dct.get(team_1, 0)
                    draw_dct[team_2] = draw_dct.get(team_2, 0)
                    draw_dct[team_1] = draw_dct.get(team_1, 0)
                elif winning_team_index == 1:
                    team_wins_dct[team_2] = team_wins_dct.get(team_2, 0) + 2
                    team_wins_dct[team_1] = team_wins_dct.get(team_1, 0)
                    win_dct[team_2] = win_dct.get(team_2, 0) + 1
                    win_dct[team_1] = win_dct.get(team_1, 0)
                    lost_dct[team_1] = lost_dct.get(team_1, 0) + 1
                    lost_dct[team_2] = lost_dct.get(team_2, 0)
                    draw_dct[team_1] = draw_dct.get(team_1, 0)
                    draw_dct[team_2] = draw_dct.get(team_2, 0)
            except IndexError:
                print(f"{team_1} vs {team_2}\nResults of the Match: {res[0]}\nMatch Draw ⚽⚽⚽\n") 
                team_wins_dct[team_1] = team_wins_dct.get(team_1, 0) + 1
                team_wins_dct[team_2] = team_wins_dct.get(team_2, 0) + 1
                draw_dct[team_1] = draw_dct.get(team_1, 0) + 1
                draw_dct[team_2] = draw_dct.get(team_2, 0) + 1
                win_dct[team_1] = win_dct.get(team_1, 0)
                lost_dct[team_1] = lost_dct.get(team_1, 0)
                win_dct[team_2] = win_dct.get(team_2, 0)
                lost_dct[team_2] = lost_dct.get(team_2, 0)
    
    group_results = [win_dct, draw_dct, lost_dct, team_wins_dct, goal_scored_dct, goal_against_dct]
    Group_standings[grp_name] = group_results


# In[141]:


#Display group stage results

for grp_name in list(Group_standings.keys()):

    team_wins_dct= dict(sorted(Group_standings[grp_name][3].items()))
    goal_scored_dct=dict(sorted(Group_standings[grp_name][4].items()))
    goal_against_dct=dict(sorted(Group_standings[grp_name][5].items()))
    
    win_dct=dict(sorted(Group_standings[grp_name][0].items()))
    draw_dct=dict(sorted(Group_standings[grp_name][1].items()))
    lost_dct=dict(sorted(Group_standings[grp_name][2].items()))
    
    lst_teams=list(team_wins_dct.keys())
    
    win_lst=list(win_dct.values())
    draw_lst=list(draw_dct.values())
    lost_lst=list(lost_dct.values())
    
    lst_win_count=list(team_wins_dct.values())
    goal_scored=list(goal_scored_dct.values())
    goal_against=list(goal_against_dct.values())
    goal_differance=[goal_scored[i]-goal_against[i] for i in range (len(goal_scored))]
    ranking_table=pd.DataFrame(list(zip(lst_teams,win_lst,draw_lst,lost_lst,goal_scored,goal_against,goal_differance,lst_win_count)),columns=["Team","Wins","Draw","Lost","Goal Scored","Goal Against","Goal Differance","Points"])
    ranking_table=ranking_table.sort_values("Points",ascending=False).reset_index(drop=True)
    ranking_table.index = ranking_table.index + 1
    print(f"\n\n{grp_name} Final Rankings")
    print(ranking_table.to_markdown())
    


# In[142]:


##Round of 16 Section_1

qualified_teams_1=[]
standings=list(Group_standings.keys())
i=0
print(f"Round of 16\n")
while i < (len(standings)):
    A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
    team_1=A_team[0][0]
    B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
    team_2=B_team[1][0]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n  {team_lst[win]} wins 🏆🏆")
            print(f"    {team_lst[win]} into the Quater-Finals ⏩⏩ \n")
            qualified_teams_1.append(team_lst[win])
    except IndexError:
            print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
            winning_team=random.choice(team_lst)
            print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
            print(f"    {winning_team} into the Quater-Finals ⏩⏩ \n")
            qualified_teams_1.append(winning_team)
    i=i+2
    
##Round of 16 Section_2
qualified_teams_2=[]
standings=list(Group_standings.keys())
i=0
while i < (len(standings)):
    A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
    team_1=A_team[1][0]
    B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
    team_2=B_team[0][0]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n  {team_lst[win]} wins 🏆🏆")
            print(f"    {team_lst[win]} into the Quater-Finals ⏩⏩ \n")
            qualified_teams_2.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
            winning_team=random.choice(team_lst)
            print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
            print(f"    {winning_team} into the Quater-Finals ⏩⏩ \n")
            qualified_teams_2.append(winning_team)
    i=i+2


# In[ ]:


print(f"Teams selected to the Quater Finals - {qualified_teams_1+qualified_teams_2}")


# In[ ]:


#Quarter Finals

Semifinal_teams=[]
i=0
print(f"Quater Final Matches\n")
while i < (len(qualified_teams_1))-1:
    team_1= qualified_teams_1[i]
    team_2= qualified_teams_1[i+1]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n  {team_lst[win]} wins 🏆🏆")
            print(f"    {team_lst[win]} into the Semi-Finals ⏩⏩ \n")
            Semifinal_teams.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽")
            winning_team=random.choice(team_lst)
            print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
            print(f"    {winning_team} into the Semi-Finals ⏩⏩ \n")
            Semifinal_teams.append(winning_team)
    i=i+2
    
i=0
while i < (len(qualified_teams_2))-1:
    team_1= qualified_teams_2[i]
    team_2= qualified_teams_2[i+1]
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n  {team_lst[win]} wins 🏆🏆")
            print(f"    {team_lst[win]} into the Semi-Finals ⏩⏩ \n")
            Semifinal_teams.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
            winning_team=random.choice(team_lst)
            print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
            print(f"    {winning_team} into the Semi-Finals ⏩⏩ \n")
            Semifinal_teams.append(winning_team)
    i=i+2


# In[ ]:


print(f"Teams selected to the Semi-Finals - {Semifinal_teams}")


# In[ ]:


#Semi Finals

final_teams=[]
third_place_match_teams=[]
i=0
print(f"Semi Final Matches\n")
while i < (len(Semifinal_teams))-1:
    team_1= Semifinal_teams[i]
    team_2= Semifinal_teams[i+1]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n  {team_lst[win]} wins 🏆🏆")
            print(f"    {team_lst[win]} into the FiIFA-Finals ⏩⏩ \n")
            final_teams.append(team_lst[win])
            third_place_match_teams.append(team_lst[(win+1)%2])

            
    except IndexError:
            print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
            winning_team=random.choice(team_lst)
            print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
            print(f"    {winning_team} into the FIFA-Finals ⏩⏩ \n")
            final_teams.append(winning_team)
            team_lst.remove(winning_team)
            third_place_match_teams.append(team_lst[0])
    i=i+2
    


# In[ ]:


print(f"Teams selected to the Finals - {final_teams}")
print(f"Teams selected to the Third Place match - {third_place_match_teams}")


# In[ ]:


#Finals and Third Place match

print(f"FiIFA FINAl\n")
team_1= final_teams[1]
team_2= final_teams[0]
    
team_1_num=label_encoder.transform([team_1])[0]
team_2_num=label_encoder.transform([team_2])[0]
team_lst=[team_1,team_2]
    
Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
res=model.predict(Input_vector)
win,_=select_winning_team(res)

try:
    print(f"{team_1} vs {team_2} \n  {team_lst[win]} are the Winners 🎉🏆🎉\n\n")
    winner=team_lst[win]
    place_2=team_lst[(win+1)%2]
            
except IndexError:
    print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
    winning_team=random.choice(team_lst)
    print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
    print(f"    {winning_team} are the Winners 🎉🏆🎉\n\n")
    winner=winning_team
    
    team_lst.remove(winning_team)
    place_2=team_lst[0]

print(f"Third Place match\n")
team_1= third_place_match_teams[1]
team_2= third_place_match_teams[0]
    
team_1_num=label_encoder.transform([team_1])[0]
team_2_num=label_encoder.transform([team_2])[0]
team_lst=[team_1,team_2]
    
Input_vector=np.array([[year,host_num,team_1_num,team_2_num]])
res=model.predict(Input_vector)
win,_=select_winning_team(res)

try:
    print(f"{team_1} vs {team_2} \n  {team_lst[win]} Wins the 3rd Place 🎉🏆🎉\n")
    place_3=team_lst[win]
            
except IndexError:
    print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽") 
    winning_team=random.choice(team_lst)
    print(f"   {winning_team} wins at Penaly Shoot-Out 🥅🎯")
    print(f"    {winning_team} Wins the 3rd Place 🎉🏆🎉\n")
    place_3=winning_team
    

    
print(f"\n\nWinner is {winner} 🥇🥇🥇")
print(f"Runner-up is {place_2} 🥈🥈🥈")
print(f"3rd Place is {place_3} 🥉🥉🥉")


# <img src= "https://images.indianexpress.com/2020/10/fifa-world-cup-trophy.jpg" alt ="Wining Moment">

# In[ ]:




