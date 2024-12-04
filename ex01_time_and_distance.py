#%% Import

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Load dataset

# Read to dataframe
df = pd.read_csv('time_dist.csv', delimiter=',')

# Characteristics
df.info()

# Statistics
df.describe()

#%% Grafico de dispersão com ajuste linear

plt.figure(figsize=(15,10))

sns.regplot(data = df,
            x = 'distance',
            y = 'time',
            marker = 'o',
            ci = False,
            scatter_kws = {'color':'navy', 'alpha':0.9, 's':220},
            line_kws = {'color':'gray', 'linewidth': 5})
