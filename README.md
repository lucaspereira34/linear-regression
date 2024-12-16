# Linear Regression

**Linear Regression** is a **supervised learning** technique that uses labeled data to learn the relationship between input explanatory variables, which are assumed to be linearly related, and a quantitative dependent variable. As a **supervised learning** method, it can be used for predictive purposes.

# Simple Linear Regression

Models the relationship between a single **explanatory variable** (X) and a **dependent variable** (Y).

$$
Y = \alpha + \beta.X + \mu
$$

- $\alpha$ is the **intercept**, the starting point of the regression line on the Y-axis when X is zero.
- $\beta$ is the **slope** coefficient, which represents the steepness and direction of the line.
- $\mu$ is the **error** term.

## Example: Time and Distance

For this example, we apply a linear regression model to the relationship between the distance a student walks to school and the time it takes to complete the walk.

Since it has only one explanatory variable (distance), the model is classified as a Simple Linear Regression.

The dataset is stored in the *time_dist.csv* file, and the Python script for this example is *ex01_time_and_distance.py*.

<img src="https://github.com/user-attachments/assets/888b6e53-30d0-4f75-b8f6-f2ad0f17cf1f" alt="Linear Regression Model Fit" widht="500" height="200">



