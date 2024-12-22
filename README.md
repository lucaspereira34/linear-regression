# Linear Regression

**Linear Regression** is a **supervised learning** technique that uses labeled data to learn the relationship between input explanatory variables, which are assumed to be linearly related, and a quantitative dependent variable. As a **supervised learning** method, it can be used for predictive purposes.

# Simple Linear Regression

Models the relationship between a single **explanatory variable** (X) and a **dependent variable** (Y).

$$
Y = \alpha + \beta.X + \mu
$$

- $\alpha$ is the **intercept** coefficient. It is the starting point of the regression line on the Y-axis when X is zero.
- $\beta$ is the **slope** coefficient. It represents the steepness and direction of the line.
- $\mu$ is the **residual** term. Also called error, it is the difference between observed values ($Y$) and predicted values ($\hat{Y}$).

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

<img src="https://github.com/user-attachments/assets/6ea9057a-62ce-4c29-874b-5d7d2d023672" alt = "Dataset info">
<br>
<img src="https://github.com/user-attachments/assets/206b7c02-5894-4b34-a3b4-20e3edf42b30" alt ="Dataset statistics">



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

### Ordinary Least Squares (OLS)

The most common method for estimating the coefficients of a linear regression model is **Ordinary Least Squares (OLS)**. It consists in determining the coefficients $\alpha$ and $\beta$ that **minimize** the sum of the squared **residuals**, so-called **Residual Sum of Squares (RSS)**:

$$
RSS = \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
$$

The mathematical properties of the OLS method are:

- The sum of the residuals equals zero: $\sum_{i=1}^{n}e_i=0$
- Residual Sum of Squares is minimized: $\min_{\alpha,\beta} \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$

The *OLS.from_formula* function from *statsmodels.api* library returns a regression model instance.

~~~python
# Estimation of the model
model = sm.OLS.from_formula('time ~ distance', df).fit()
~~~

