# Linear Regression

**Linear Regression** is a **supervised learning** technique that uses labeled data to learn the relationship between input explanatory variables, which are assumed to be linearly related, and a quantitative dependent variable. As a **supervised learning** method, it can be used for predictive purposes.

# Simple Linear Regression

Models the relationship between a single **explanatory variable** and a **dependent variable**.

$$
\hat{Y} = \alpha + \beta.X
$$

- $\hat{Y}$ represents the **predicted values** of the **dependent variable**, also called **fitted values**.
- $\alpha$ is the **intercept** coefficient. It is the point where the regression line intersects the Y-axis when X equals zero.
- $\beta$ is the **slope** coefficient. It represents the steepness and direction of the regression line.
- X is the **explanatory variable**.

$$
Y = \alpha + \beta.X + \mu
$$

- Y represents the **observed values** of the **dependent variable**.
- $\mu$ is the **residual** term, also called error term. It is the difference between the observed values $Y$ and the predicted values $\hat{Y}$.

## Example: Time and Distance

For this example, we apply a linear regression model to the relationship between the distance a student walks to school and the time it takes to complete the walk.

Since it has only one explanatory variable (distance), the model is classified as a Simple Linear Regression.

- Dataset: *time_dist.csv*
- Python script: *ex01_time_and_distance.py*

~~~python
# Read dataset
df = pd.read_csv('time_dist.csv', delimiter=',')

# Characteristics
df.info()

# Statistics
df.describe()
~~~

There are 2 columns with 10 observations:

<img src="https://github.com/user-attachments/assets/6ea9057a-62ce-4c29-874b-5d7d2d023672" alt = "Dataset info" width="300" height="90">
<br>
<img src="https://github.com/user-attachments/assets/206b7c02-5894-4b34-a3b4-20e3edf42b30" alt ="Dataset statistics" width="200" height="150">



### Scatter Plot and Linear Regression

The scatter plot in **Plot 1** provides a clear visualization of the data distribution. The points follow a linear trend, suggesting that a linear regression model could effectively capture the relationship between the variables, as demonstrated in **Plot 2**.

<img src="https://github.com/user-attachments/assets/c7e87abb-2450-493e-a162-86eab47c5c85" alt = "Time_dist Scatter plot" width="300" height="200">

<img src="https://github.com/user-attachments/assets/1f4aa500-d350-4126-b779-5362787c940b" alt="Linear Regression Model Fit" widht="500" height="200">
<br><br>

~~~python
# Plot 1: Basic scatter plot
plt.figure(figsize=(15,10))
sns.scatterplot(data = df, x = 'distance', y = 'time', color = 'navy', alpha = 0.9, s = 220)
plt.xlabel('distance', fontsize=28)
plt.ylabel('time', fontsize=28)
plt.tick_params(axis='both', labelsize=24)
plt.show()

# Plot 2: Scatter plot with linear regression fit
plt.figure(figsize=(15,10))
sns.regplot(data = df, x = 'distance', y = 'time', marker = 'o', ci = False,
            scatter_kws = {'color':'navy', 'alpha':0.9, 's':220},
            line_kws = {'color':'gray', 'linewidth': 5})
plt.xlabel('distance', fontsize=28)
plt.ylabel('time', fontsize=28)
plt.tick_params(axis='both', labelsize=24)
plt.show()
~~~

### Simple OLS Regression

The most common method for estimating the coefficients of a linear regression model is **Ordinary Least Squares (OLS)**. It consists in determining the coefficients $\alpha$ and $\beta$ that **minimize** the sum of the squared **residuals**, so-called **Residual Sum of Squares (RSS)**:

$$
RSS = \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
$$

The mathematical properties of the OLS method are:

- The sum of the residuals equals zero:

$$
\sum_{i=1}^{n}e_i=0
$$

- Residual Sum of Squares is minimized:

$$
\min_{\alpha,\beta} \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
$$

The *OLS.from_formula* method from the *statsmodels.api* library returns a regression model instance. The *.fit()* method is then used to fit the model and return the *results* instance.

~~~python
# Estimation of the model
model = sm.OLS.from_formula('time ~ distance', df)
results = model.fit()
~~~

The results are summarized using the *.summary()* method.

~~~python
results.summary()
~~~

<img src=https://github.com/user-attachments/assets/195241d7-4aa7-4399-b81e-c35dd842c0a1 alt="Simple Regression Results" width="570" height="400">

### Coefficients, Fitted Values and Residuals

The coefficients of the model can be obtained from the *results* instance, either through the *.summary()* output or by using the *.params* attribute:

~~~python
intercept = results.params['Intercept']
slope = results.params['distance']

print('Intercept: ', intercept.round(2))
print('Slope: ', slope.round(2))
~~~

<img src="https://github.com/user-attachments/assets/ef3bfc74-1a5b-41e6-9a15-aa3787dc9b2c" alt="Print coefficients" width="140" height="40">


Applying these coefficients to the formula, the fitted model can be expressed by:

$$
\hat{Y} = 5.88 + 1.42.X
$$

And the residuals by:

$$
\mu = Y - \hat{Y}
$$

The fitted values and the residuals can be directly retrieved from the *results* instance using the attributes *.fittedvalues* and *.resid*, respectively.

~~~python
# Store fitted values and residuals
df['y_fitted'] = results.fittedvalues
df['y_resid'] = results.resid
~~~

### Goodness of Fit

The **godness of fit** indicates the percentage of variance in the **dependant variable** Y that is explained by the joint variation of the **explanatory variables** X. It ranges from 0 to 1, and **the higher the coefficient, the greater the predictive power of the regression model**.

It is commonly measured using the **R-Squared ($R^2$)** formula.

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

- RSS &rarr; [Residual Sum of Squares](#simple-ols-regression)
- TSS &rarr; **Total Sum of Squares**: The sum of the squared differences between the observed values $Y_i$ and the mean of the observed values $\bar{Y}$

$$
TSS = \sum{(Y_i - \bar{Y})^2}
$$

Although the R-Squared can be calculated through its formula, it can also be directly obtained from the *results* instance through the *.summary()* output or the *.rsquared* attribute.

~~~python
r_2 = results.rsquared
print('R²: ', r_2.round(4))
~~~

<img src="https://github.com/user-attachments/assets/6dcb1ba1-0915-4d5f-9aa5-e0ff09a873d3" alt="Print r2" width="100" height="20">

**Plot 3** shows the graphical representation of **RSS** with green dotted lines, and **TSS** with purple dashed lines.

<img src="https://github.com/user-attachments/assets/7021cab5-84a5-4e32-94b3-2b5409a00949" alt="Plot R-squared concept" width="400" height="300">
<br><br>

~~~python
# Plot 3: Concept of R²
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
    plt.axhline(y = y.mean(), color = 'silver', line style = '-', linewidth=4)
    plt.title('R²: ' + str(round(results.rsquared, 4)), fontsize=30)
    plt.xlabel('Distance', fontsize=24)
    plt.ylabel('Time', fontsize=24)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    plt.xlim(0, 35)
    plt.ylim(0, 60)
    plt.legend(['Fitted Values', 'Y_fitted - Y_average', 'Residual = Y - Y_fitted'],
               fontsize=22, loc='upper left')
plt.show()
~~~

### Confidence Interval

Confidence intervals (CI) provide a range of plausible values for the **coefficients** with a given **confidence level**. For example, a 95% CI means that in a scenario where we repeatedly sample and estimate the model, 95% of the resulting confidence intervals would contain the **true population coefficients**, based on Student's t-distribution. 

The *.summary()* method includes the 95% confidence intervals for the intercept and slopes. Specific intervals can be retrieved using the *.conf_int()* method, which uses the **significance level**.

- **Significance level** = 1 - **confidence level**.

~~~python
# 10% significance level / 90% confidence level
results.conf_int(alpha=0.1)

# 5% significance level / 95% confidence level
results.conf_int(alpha=0.05)

# 1% significance level / 99% confidence level
results.conf_int(alpha=0.01)
~~~

**Plots 4, 5, and 6** illustrate the confidence intervals in a simple regression fit. 

~~~python
# Confidence interval plots
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
~~~

### Predictions

As shown in the output of *df.describe()*, the dataset used to train the model has a minimum distance value of 5 km and a maximum value of 32 km. Predictions made within the range of the explanatory variables observed during model training are referred to as **Interpolation**. It's a reliable zone, where predictions are usually more trustworthy.

On the other hand, predictions made outside the observed range of the explanatory variables are referred to as **Extrapolation**. These predictions might lead to unreliable outcomes, because the model assumes that the same linear relationship holds beyond the observed data, which might not always be true.

This means that predictions made within the range of 5 km and 32 km fall into the **Interpolation** zone and are, therefore, more reliable. The model can be used to predict values for new distances within this range. While it's possible to manually calculate predictions using the estimated coefficients, it's more efficient to use the *.predict()* method.

~~~python
# How long would it take for a student to walk 21 km?

# Manual calculation using the estimated parameters
results.params[0] + results.params[1] * (21)

# Using a pandas.DataFrame inside the .predict() method
results.predict(pd.DataFrame({'distance':[21]}))
~~~

