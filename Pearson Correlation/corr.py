#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataKind Datadrive MAY2019
"""

# Check all packages installed in current env before proceeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import os

# Set project directory !! INPUT REQUIRED !!
project_dir = os.path.join("/","Users","MT","Nextcloud","Projects","ICT4D-DEU20GIZ6926 ProSoil GLOBAL","Testing","PSG","data")




# Process dataset
#______________________________________________________________________________

# Set path to file and import datset
data = "sqllab_untitled_query_2_20210308T120821.csv"
path = os.path.join(project_dir,data)
df = pd.read_csv(path)

# List of interesting variables
col_names_filter = (
    'cp_name',
    'reporting_year',
    'data_type',
    'm1_0_0_rehabilitated_ha',
    'm1_0_1_rehabilitated_onfarm_ha',
    'm1_0_2_rehabilitated_offfarm_ha',
    'm2_0_0_women_improved_pc',
    'm2_0_1_women_improved_hh_n',
    'm2_0_2_women_improved_n',
    'm3_0_3_crops_increase_avg',
    'm4_0_0_incentives_n',
    'o1_1_0_adopting_farm_pc',
    'o1_1_1_adopting_farm_n',
    'o1_1_2_adopting_farm_female_pc',
    'o1_1_3_adopting_farm_female_n',
    'o1_2_1_steps_total_n',
    'o1_2_2_steps_impl_n',
    'o1_2_3_steps_impl_pc',
    'o1_3_0_measures_n',
    'o1_3_1_measures_women_n',
    'o2_1_0_strategies_n',
    'o2_2_0_courses_n',
    'o3_1_1_lessons_n',
    'o3_2_1_exchanges_n',
    'o3_3_0_knowledge_n',
    'o3_3_1_knowledge_female_pc',
    'o3_3_2_knowledge_female_n',
    'z2_1_train_target_n',
    'z2_2_train_target_female_n',
    'z2_3_train_target_female_pc',
    'z2_4_train_target_young_n',
    'z2_5_train_target_young_pc',
    'z2_6_train_other_n',
    'z2_7_train_other_female_n',
    'z2_8_train_other_female_pc',
    'z2_9_train_other_young_n',
    'z2_10_train_other_young_pc',
    'z2_11_train_total_n',
    'z2_12_train_total_female_n',
    'z2_13_train_total_female_pc',
    'z2_14_target_avg_hh_n',
    'z2_15_target_avg_ben_hh_n',
    'z2_16_target_hh_calc_factor',
    'z2_17_train_direct_ben_n')


# Select relevant columns from master pollution
df_filtered = df.loc[:,col_names_filter]
df_filtered = df_filtered.query('data_type == False')
df_filtered = df_filtered.query('reporting_year == "2020"')



# # Compute Pollution-Deprivation Distance Index
# #______________________________________________________________________________

# # Select relevant columns for distance computation
# pollution_dist = pollution_filtered.loc[:,["LSOA.Code", "NOx", "IMD", "Households.pov"]]

# # Normalise and add derived variables for each of NOx, IMD and Pov
# pollution_dist["sc_nox"] = (pollution_dist["NOx"] - pollution_dist["NOx"].mean())/pollution_dist["NOx"].std()
# pollution_dist["sc_imd"] = (pollution_dist["IMD"] - pollution_dist["IMD"].mean())/pollution_dist["IMD"].std()
# pollution_dist["sc_pov"] = (pollution_dist["Households.pov"] - pollution_dist["Households.pov"].mean())/pollution_dist["Households.pov"].std()

# # Compute euclidian distance between NOx and IMD/Household poverty
# pollution_dist["dist_imd"] = (((pollution_dist["sc_nox"].min() - pollution_dist["sc_nox"])**2 + 
#               (pollution_dist["sc_imd"].min() - pollution_dist["sc_imd"])**2)**0.5)
# pollution_dist["dist_pov"] = (((pollution_dist["sc_nox"].min() - pollution_dist["sc_nox"])**2 + 
#               (pollution_dist["sc_pov"].min() - pollution_dist["sc_pov"])**2)**0.5)

# # Standardise range for each index variable between 0 and 1
# pollution_dist["dist_imd_01"] = pollution_dist["dist_imd"]/pollution_dist["dist_imd"].max()
# pollution_dist["dist_pov_01"] = pollution_dist["dist_pov"]/pollution_dist["dist_pov"].max()
              
# # Output to csv
# pollution_filtered.to_csv(os.path.join(project_dir,"pollution_dist.csv"))



# Correlation Matrix as Heatmap
#______________________________________________________________________________

# Remove row numbers and LSOA columns
df_red = df_filtered.iloc[:,3:] 

##### HEATMAP
corr = np.round(df_red.corr(),2) # compute correlation matrix and round 2
mask = np.zeros_like(corr) # Set mask to show only bottom half of corr matrix
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    plt.subplots(figsize = (30, 24))
    cmap = sns.diverging_palette(220, 10, n=20)
    ax = sns.heatmap(corr, cmap=cmap,mask=mask, vmax=0.5, vmin=-0.5, square=True, annot = True, annot_kws = {'fontsize': 12}, cbar_kws={'shrink': .8})

# Output file
ax.figure.savefig(os.path.join(project_dir,"psg_corr.png"))







