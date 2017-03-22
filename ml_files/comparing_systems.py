
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# In[3]:

janus_timefile = '../data/janus/janus_unprocessed_timings.csv'
bridges_timefile = '../data/bridges/bridges_unprocessed_timings.csv'
comet_timefile = '../data/comet/comet_unprocessed_timings.csv'
properties_file = '../data/unprocessed_properties.csv'

janus_times = pd.read_csv(janus_timefile, header=0, index_col=0)
bridges_times = pd.read_csv(bridges_timefile, header=0, index_col=0)
comet_times = pd.read_csv(comet_timefile, header=0, index_col=0)
properties = pd.read_csv(properties_file, header=0, index_col=0)

times = [janus_times, bridges_times, comet_times]
combined_times = pd.concat(times)
combined_times.dropna()
combined_times.drop_duplicates()
combined_times.info()
combined_times.describe()


# In[48]:

converged = combined_times[combined_times['status_id'] == 1]
grouped = converged.groupby(['system_id', 'numprocs', 'matrix'])
grouped.describe()


# In[63]:

i = 0
best_times = grouped.new_time.aggregate(np.min)
for sys_np_mat, group in grouped:
    print(sys_np_mat, best_times[sys_np_mat])
    for t in group.new_time:
        if t <= 1.25 * best_times[sys_np_mat]:
            print('GOOD ', t) 
        #print('b', best_times[sys_np_mat])
        
    i += 1
    if i is 5:
        sys.exit()
    


# In[54]:

i_a = 0
X_a_train, X_a_test = [], []
X_b_train, X_b_test = [], []
y_a_train, y_a_test = [], []
y_b_train, y_b_test = [], []

np_a = 1
np_b = 1
system_a = [1,2]
system_b = 0

a = pd.DataFrame()
b = pd.DataFrame()

if type(np_a) == str and np_a == "all":
    a = all_combined
elif type(np_a) == int:
    a = all_combined[(all_combined.np == np_a)]
elif type(np_a) == list:
    for num in np_a:
        a = a.append(all_combined[(all_combined.np == num)], 
                     ignore_index=True)

# now process system_a
if type(system_a) == str and system_a == "all":
    a = a
elif type(system_a) == int:
    a = a[(a.system_id == system_a)]
elif type(system_a) == list:
    temp = pd.DataFrame()
    for num in system_a:
        temp = temp.append(a[(a.system_id == num)], ignore_index=True)
    a = temp

a.describe()
l = list(a.columns)
l[-1]

