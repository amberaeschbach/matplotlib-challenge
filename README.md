# matplotlib-challenge
module 5 homework
#import dependencies
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from pathlib import Path
import numpy as np
from scipy.stats import linregress
#load in csvs
mouse_metadata_path = Path("Data/Mouse_metadata.csv")
study_results_path = Path("Data/Study_results.csv")
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)
#merge csv
mouse_data_complete = pd.merge(study_results,mouse_metadata,on="Mouse ID",how="left")
mouse_data_complete.head()
#checking number of mice
mice_num = mouse_data_complete["Mouse ID"].nunique()
mice_num
249
#getting the duplicate mice ID number that shows up for mouse ID and timepoint
dup_mice = mouse_data_complete.loc[mouse_data_complete.duplicated(subset=["Mouse ID", "Timepoint"]),"Mouse ID"].unique
dup_mice
<bound method Series.unique of 137     g989
360     g989
681     g989
869     g989
1111    g989
Name: Mouse ID, dtype: object>
#optional: get all the data for the duplicate mouse ID
dup_mice_df = mouse_data_complete.loc[mouse_data_complete["Mouse ID"] == "g989",:]
dup_mice_df
#create a clean dataframe by dropping the duplicate mouse by its ID
clean_df= mouse_data_complete[mouse_data_complete['Mouse ID'] != "g989"]
clean_df.head()
#checking the number of mice in the clean dataframe
clean_mice_num = clean_df["Mouse ID"].nunique()
clean_mice_num
#generate a summary  statistics table of mean, median, standard deviation, and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# I GOT HELP FROM CHATGPT HERE TO DO THE GROUPBYS
mean = clean_df['Tumor Volume (mm3)'].groupby(clean_df['Drug Regimen']).mean()
median = clean_df['Tumor Volume (mm3)'].groupby(clean_df['Drug Regimen']).median()
var = clean_df['Tumor Volume (mm3)'].groupby(clean_df['Drug Regimen']).var()
std = clean_df['Tumor Volume (mm3)'].groupby(clean_df['Drug Regimen']).std()
sem = clean_df['Tumor Volume (mm3)'].groupby(clean_df['Drug Regimen']).sem()

summary_stat = pd.DataFrame({"Mean Tumor Volume":mean, 
                            "Median Tumor Volume":median, 
                           "Tumor Volume Variance":var, 
                           "Tumor Volume Std. Dev.":std, 
                           "Tumor Volume Std. Err.":sem})
# Display the Summary statistics table grouped by 'Drug Regimen' column
summary_stat
#generate a summary statistics table of mean, median, variance, standard deviation, and sem of the tumor volume for each region
#using the aggregation method, produce the same summary statistics in a single line
agg_sum =  clean_df.groupby(['Drug Regimen'])[['Tumor Volume (mm3)']].agg(['mean', 'median', 'var', 'std', 'sem'])
agg_sum
#getting mice count for bar plot
mice_count = clean_df["Drug Regimen"].value_counts()
mice_count
#generate a bar plot showing the total number of timepoints for all mice tested for each drug regimen using pandas
pandas_plot = mice_count.plot.bar(color='b')
plt.xlabel("Drug Regimen")
plt.ylabel("Number of Mice Tested")
#HERE IS THE FIRST PLACE IN MY CODE WHERE I GOT HELP FROM A PEER, TO FIND THE BAR PLOT
#generate a bar plot showing the total number of timepoints for all mice tested for each drug regimen using pyplot.
x_axis = mice_count.index.values
y_axis = mice_count.values
plt.bar(x_axis, y_axis, color='b', align='center')
plt.xlabel("Drug Regimen")
plt.ylabel("Number of Mice Tested")
plt.xticks(rotation="vertical")
plt.show()
#generate a pie plot showing the distribution of female versus male mice using pandas
#I USED CHATGBT TO LOOK UP HOW TO FORMAT THE NUMBERS HERE
male_v_female = clean_df["Sex"].value_counts()
male_v_female.plot.pie(autopct= "%1.1f%%")
# generate a pie plot showing the distribution of female versus male mice using pyplot
labels = ['Female', 'Male']
sizes = [49.7999197, 50.200803]
plot = male_v_female.plot.pie(y='Total Count', autopct="%1.1f%%")
plt.ylabel('Sex')
plt.show()
#I GOT HELP FROM MY TUTOR HERE WITH THIS CONCEPT OF THE MERGE AND GROUPBY
timepoint_max = clean_df.groupby("Mouse ID")["Timepoint"].max()
timepoint_max = timepoint_max.reset_index()
timepoint_merge = timepoint_max.merge(clean_df,on=["Mouse ID","Timepoint"])
timepoint_merge
plt.show()
#MY TUTOR HELPED ME PUT TOGETHER THIS FOR LOOP
#put treatments into a list for for loop (and later for plot labels)
treatment_list = ["Capomulin", "Ramicane", "Infubinol","Ceftamin"]

tumor_vol_list = []

for drug in treatment_list:
    tumor_volume = timepoint_merge.loc[timepoint_merge["Drug Regimen"]==drug,"Tumor Volume (mm3)"]
    tumor_vol_list.append(tumor_volume)
    
    tumor_quantiles = tumor_volume.quantile([.25,.5,.75])
    lowerq = tumor_volume.quantile(.25)
    upperq = tumor_volume.quantile(.75)
    iqr = upperq - lowerq
    lowerb = lowerq - (1.5*iqr)
    upperb = upperq + (1.5*iqr)
    outliers = tumor_volume.loc[(tumor_volume > upperb) | (tumor_volume < lowerb)]
    print(drug,outliers)
#AN LA HELPED ME BUILD THIS BOXPLOT
#generate a box plot that shows the distribution of the tumor volume for each treatment group
plt.boxplot(tumor_vol_list,labels=treatment_list)
plt.ylabel("Final Tumor Volume (mm3)")
#generate a line plot of tumor volume vs timepoint for a mouse treated with capomulin
line_df = clean_df.loc[clean_df["Mouse ID"] == "l509",:]
line_df.head()
x_axis = line_df["Timepoint"]
tumersize = line_df["Tumor Volume (mm3)"]

fig1, ax1 = plt.subplots()
plt.title('Capomulin treatmeant of mouse l509')
plt.plot(x_axis, tumersize,linewidth=2, markersize=15,color="blue")
plt.xlabel('Timepoint (Days)')
plt.ylabel('Tumor Volume (mm3)')
plt.show()
#AN LA HELPED ME WITH THIS SCATTERPLOT
#generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
fig1, ax1 = plt.subplots()

avg_vol_weight = clean_df.loc[clean_df["Drug Regimen"] == "Capomulin",:]
avg_vol_weight_plot = avg_vol_weight.groupby("Mouse ID")[["Weight (g)","Tumor Volume (mm3)"]].mean()


marker_size=15
plt.scatter(avg_vol_weight_plot["Weight (g)"],avg_vol_weight_plot['Tumor Volume (mm3)'],s=175, color="blue")
plt.xlabel('Weight (g)',fontsize =14)
plt.ylabel('Average Tumor Volume (mm3)')
plt.show()
#calculate the correlation coefficient and linear regression model 
#for mouse weight and average tumor volume for the Capomulin regimen
#I USED CHAT GPT TO HELP ME FIND THE CORRELATION AND THE SCATTERPLOT WITHT THE REGRESSION LINE
correlation = st.pearsonr(avg_vol_weight_plot['Weight (g)'],avg_vol_weight_plot['Tumor Volume (mm3)'])
print(f"The correlation between mouse weight and the average tumor volume is {round(correlation[0],2)}")
(slope, intercept,rvalue, pvalue, stderr)= linregress(avg_vol_weight_plot["Weight (g)"],avg_vol_weight_plot["Tumor Volume (mm3)"])
regress_values=avg_vol_weight_plot["Weight (g)"]* slope + intercept
line_eq= f"y = {round(slope, 2)} x + {round(intercept, 2)}"

plt.scatter(avg_vol_weight_plot["Weight (g)"],avg_vol_weight_plot["Tumor Volume (mm3)"],color='b')
plt.plot(avg_vol_weight_plot["Weight (g)"], regress_values, color='red')
plt.xlabel("Weight (g)")
plt.ylabel("Average Tumor Volume (mm3)")
plt.show()
plt.show()
