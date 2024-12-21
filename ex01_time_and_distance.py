# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install playsound
!pip install pingouin
!pip install emojis
!pip install statstests

#%% Import

import pandas as pd # data wrangling
import numpy as np # math and array operations
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
import plotly.graph_objects as go # 3D plots
from scipy.stats import pearsonr # Pearson correlation
import statsmodels.api as sm # model fitting
from statsmodels.iolib.summary2 import summary_col # compare models
from sklearn.preprocessing import LabelEncoder # data transformation
from playsound import playsound # sound reproduction
import pingouin as pg # correlation matrix
import emojis # emojis for plots
from statstests.process import stepwise # Stepwise regression
from statstests.tests import shapiro_francia # Shapiro-Francia test
from scipy.stats import boxcox # Box-Cox transformation
from scipy.stats import norm # normal plot
from scipy import stats
import webbrowser

#%% Load dataset

# Read to dataframe
df = pd.read_csv('time_dist.csv', delimiter=',')

# Characteristics
df.info()

# Statistics
df.describe()

#%% Basic Scatter plot

plt.figure(figsize=(15,10))

sns.scatterplot(data = df,
            x = 'distance',
            y = 'time',
            color = 'navy',
            alpha = 0.9,
            s = 220)

plt.title('Plot 1: Basic scatter plot', fontsize=28)
plt.xlabel('distance', fontsize=28)
plt.ylabel('time', fontsize=28)
plt.tick_params(axis='both', labelsize=24)
plt.show()

#%% Scatter plot with linear regression fit

plt.figure(figsize=(15,10))

sns.regplot(data = df,
            x = 'distance',
            y = 'time',
            marker = 'o',
            ci = False,
            scatter_kws = {'color':'navy', 'alpha':0.9, 's':220},
            line_kws = {'color':'gray', 'linewidth': 5})

plt.title('Plot 2: Linear regression fit', fontsize=28)
plt.xlabel('distance', fontsize=28)
plt.ylabel('time', fontsize=28)
plt.tick_params(axis='both', labelsize=24)
plt.show()

#%% Interactive scatter plot ('EXAMPLE01.html' figure)

# Data
x = df.distance
y = df.time

# Linear regression
slope, intercept = np.polyfit(x, y, 1)
y_trend = slope * x + intercept

fig = go.Figure()

# Real values
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(color='navy', size=20), name='Valores Reais')
    )

# Fitted values
fig.add_trace(go.Scatter(
    x=x,
    y=y_trend,
    mode='lines',
    line=dict(color='dimgray', width=5), name='Fitted Values')
    )

# Layout
fig.update_layout(
    xaxis_title='Distance',
    yaxis_title='Time',
    title={
        'text': 'Scatter Plot with Fitted Values',
        'font': {'size': 20, 'color': 'black', 'family': 'Arial'},
        'x': 0.5,
        'y': 0.97,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='snow',
    xaxis=dict(gridcolor='black'),
    yaxis=dict(gridcolor='black'),
    showlegend=True
)

fig.write_html('EXAMPLE01.html')

# Abrir o arquivo HTML no navegador
webbrowser.open('EXAMPLE01.html')

#%% Estimação do modelo de regressão linear simples

# Estimação do modelo
model = sm.OLS.from_formula('time ~ distance', df).fit()

# Observação dos parâmetros resultantes da estimação
model.summary()

#%% Store fitted values and residuals

df['y_fitted'] = model.fittedvalues
df['y_resid'] = model.resid

#%% Plotting the concept of R²

plt.figure(figsize=(15,10))
y = df['time']
yhat = df['y_fitted']
x = df['distance']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot(x, yhat, color='grey', linewidth=7)
    plt.plot([x[i], x[i]], [yhat[i], mean[i]], '--', color='darkorchid', linewidth=5)
    plt.plot([x[i], x[i]], [yhat[i], y[i]],':', color='limegreen', linewidth=5)
    plt.scatter(x, y, color='navy', s=220, alpha=0.2)
    plt.axhline(y = y.mean(), color = 'silver', linestyle = '-', linewidth=4)
    plt.title('R²: ' + str(round(model.rsquared, 4)), fontsize=30)
    plt.xlabel('Distance', fontsize=24)
    plt.ylabel('Time', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 35)
    plt.ylim(0, 60)
    plt.legend(['Fitted Values', 'Y_fitted - Y_average', 'Residual = Y - Y_fitted'],
               fontsize=22, loc='upper left')
plt.show()