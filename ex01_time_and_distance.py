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

# Read dataset
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

# Open HTML file on bronwser
webbrowser.open('EXAMPLE01.html')

#%% Estimação do modelo de regressão linear simples

# Estimation of the model
model = sm.OLS.from_formula('time ~ distance', df)
results = model.fit()

# Summary of the regression results
results.summary()

#%% Model coefficients 

intercept = results.params['Intercept']
slope = results.params['distance']

print('Intercept: ', intercept.round(2))
print('Slope: ', slope.round(2))

#%% Store fitted values and residuals

df['y_fitted'] = results.fittedvalues
df['y_resid'] = results.resid

#%% Goodness of fit R²

r_2 = results.rsquared
print('R²: ', r_2.round(4))

#%% Plot 3: Concept of R²

plt.figure(figsize=(15,10))
y = df['time']
yhat = df['y_fitted']
x = df['distance']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot(x, yhat, color='grey', linewidth=7)
    plt.plot([x[i], x[i]], [yhat[i], y[i]],':', color='limegreen', linewidth=5)
    plt.plot([x[i], x[i]], [yhat[i], mean[i]], '--', color='darkorchid', linewidth=5)
    plt.scatter(x, y, color='navy', s=220, alpha=0.2)
    plt.axhline(y = y.mean(), color = 'silver', linestyle = '-', linewidth=4)
    plt.title('Plot 3: R² = ' + str(round(results.rsquared, 4)), fontsize=30)
    plt.xlabel('Distance', fontsize=24)
    plt.ylabel('Time', fontsize=24)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    plt.xlim(0, 35)
    plt.ylim(0, 60)
    plt.legend(['Fitted Values', 'Residual = Y - Y_fitted', 'Y_fitted - Y_average'],
               fontsize=22, loc='upper left')
plt.show()

#%% Confidence interval

# 10% significance level / 90% confidence level
results.conf_int(alpha=0.1)

# 5% significance level / 95% confidence level
results.conf_int(alpha=0.05)

# 1% significance level / 99% confidence level
results.conf_int(alpha=0.01)

#%% Confidence interval plots

def plot_ci(df, ci, plot):
    plt.figure(figsize=(15,10))
    sns.regplot(data=df, x='distance', y='time', marker='o', ci=ci,
                scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
                line_kws={"color":'grey', 'linewidth': 5})
    plt.title(plot + ': CI ' + str(ci) + '%', fontsize=30)
    plt.xlabel('Distance', fontsize=24)
    plt.ylabel('Time', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 35)
    plt.ylim(0, 60)
    plt.legend(['Observed Values', 'Fitted Values', 'CI ' + str(ci) + '%'],
               fontsize=24, loc='upper left')
    plt.show
    
plot_ci(df, 90, 'Plot 4') # Plot 4
plot_ci(df, 95, 'Plot 5') # Plot 5
plot_ci(df, 99, 'Plot 6') # Plot 6

#%% Making predictions: How long would it take for a student to walk 21 km?

# Manual calculation using the estimated parameters
results.params[0] + results.params[1] * (21)

# Using a pandas.DataFrame inside the .predict() method
results.predict(pd.DataFrame({'distance':[21]}))
