# Diagnostic Plots

- A residual $(e)$ is the difference between the actual value of a dependent variable $(y)$ and the value predicted by the regression model $(\hat y)$ for a given observation
    
    $$
    e = y - \hat y
    $$
    
    - Residuals measure how far off the model’s prediction is for each data point
    
    ![image.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/image.png)
    
    - It is the vertical distance between the observed data point and the regression line
    - If the residual is positive $(y>\hat y)$, the model underestimated the actual value. If the residual is negative $(y<\hat y)$, the model overestimated the actual value.
    - In simple linear regression, the sum of all residuals is zero because the regression line minimizes the total error in a least-squares sense

### Residual vs. Fitted Plots

- A residual plot is a graph that shows:
    - Residuals $(e)$ on the y-axis
    - Against either predicted values $(\hat y)$ or independent variables $(x)$ on the x-axis
- Each point represents one observation, and its position indicates how far the model’s prediction was from the actual value
- What to look for in a residual plot?
    - Residuals should be randomly distributed around the horizontal line at zero. This suggests the model’s predictions are unbiased, and errors are purely random.
        
        ![Screenshot 2025-01-20 at 6.29.23 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_6.29.23_PM.png)
        
    - Systematic patterns (e.g., curves, clusters) suggest
        - Non-linearity: The model does not capture the true relationship.
        - Omitted variables: Missing variables that influence $y$
        
        ![Screenshot 2025-01-20 at 6.31.04 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_6.31.04_PM.png)
        
    - Residuals should show a constant spread across the x-axis (no funnel or cone shapes). This indicates the variance of errors is stable, satisfying the homoscedasticity assumption.
    - Increasing or decreasing spread suggests heteroscedasticity, meaning the model’s errors vary systematically with the independent variable or predicted value.
        
        ![Screenshot 2025-01-20 at 6.35.38 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_6.35.38_PM.png)
        
    - **Common Residual Plot Patterns and Their Interpretations**
        
        
        | Residual Plot Pattern | Possible Cause |
        | --- | --- |
        | Random scatter around 0 | Model fits well (assumptions are satisfied) |
        | Funnel shape (wider spread as x increases) | Heteroscedasticity (non-constant variance) |
        | Curve or U-shape | Non-linearity (model misses the true relationship) |
        | Outliers or clusters | Influential points or missing variables |
        | Wave-like patterns | Autocorrelation in errors |

### Scale-Location Plots

- A scale-location plot (also called spread-location plot) is a graph of:
    - Square root of the absolute value of the residuals $(\sqrt{|e|})$ on the y-axis
    - Predicted values $(\hat y)$ on the x-axis.
- Each point represents an observation, showing how the spread (or scale) of the residuals changes across the range of predicted values.
- The goal is to check whether the residuals are spread evenly (homoscedasticity). If the spread varies, it indicates heteroscedasticity.
- **How to Interpret a Scale-Location Plot?**
    - Ideal Case (Good Model Fit): The points in the plot are randomly scattered around a horizontal line, with no clear pattern or trend. This suggests the residuals have a constant variance across all levels of predicted values.
        
        ![Screenshot 2025-01-20 at 7.17.32 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_7.17.32_PM.png)
        
    - Signs of Heteroscedasticity:
        - Funnel Shape: The spread of the residuals increases (or decrease) as the predicted values increases. This indicates non-constant variance.
        - Pattern or Curves: Systematic patterns suggest the model might not fully capture the data structure.
        
        ![Screenshot 2025-01-20 at 7.19.46 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_7.19.46_PM.png)
        

### Q-Q Plots

- A Q-Q (quantile-quantile) plot is used to assess whether the residuals follow a normal distribution.
- It compares the quantiles of the residuals to the quantiles of a theoretical normal distribution.
    - The x-axis represents the expected quantiles of a normal distrbution
    - The y-axis represents the observed quantiles of the residuals.
    
    If the residuals are normally distributed, the points in the Q-Q plot will fall approxiamately along a 45-degree straight line. 
    
    ![Screenshot 2025-01-20 at 7.36.00 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_7.36.00_PM.png)
    
- In a Q-Q plot, quantiles from your data’s residuals are compared to the quantiles of a theoretical distribution (usually normal). This helps assess whether the residuals follow the expected distribution.
    
    For example:
    
    - The 25th percentile of your residuals is compared to the 25th percentile of a normal distribution.
    - The 50th percentile (median) is compared to the median of a normal distribution, and so on.

### Cook’s Distance Plots

- Used to identify influential data points, i.e., observations that have a disproportionately large effect on the fitted regression model.
- Cook’s Distance measures how much the regression model’s predictions change if a particular observation is removed. It combines the leverage (how far an observation’s predictor values are from the mean) and the residual size (how far the observed value is from the predicted value)
- Mathematically, Cook’s Distance for the $i$-th observation is:
    
    $$
    D_i={\text{Sum of squared changes in fitted values}\over \text{Number of predictors}\times \text{Residual variance}}
    $$
    
    A large $D_i$ indicates that observation $i$ has a significant influence on the regression model. 
    
- The goal is to identify influential data points that might unduly affect the regression model’s results. Removing or addressing these points can improve the model’s robustness and reliability.
    
    ![Screenshot 2025-01-20 at 7.52.25 PM.png](Diagnostic%20Plots%20182509a087318030a88fc5cebf2f6884/Screenshot_2025-01-20_at_7.52.25_PM.png)
    
- This plot helps us to find influential point (an observation that changes the slope of the line). We look for outlying values at the upper right corner or at the lower right corner (cases outside of the dashed lines). Those spots are the places where cases can be influential against a regression line.