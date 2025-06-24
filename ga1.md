# Graded Assignment 1

- “I have read and understood the instructions and policies for this assignment, including those
related to the use of AI and other external resources.”
- “I completed this assignment independently.”
- “I did not use AI on this assignment.”

## **Task 1: Exploratory Time Series Analysis of co2 Dataset**

![Screenshot 2025-06-23 at 2.01.36 AM.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/Screenshot_2025-06-23_at_2.01.36_AM.png)

---

### **(a): Plot and Initial Observations**

```r
# Load and plot the dataset
plot(co2, main = "Monthly Atmospheric CO2 at Mauna Loa Observatory",
     ylab = "CO2 (ppm)", xlab = "Year")
```

![image.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/image.png)

The co2 dataset consists of monthly recordings of atmospheric carbon dioxide (CO₂) levels. Plotting the dataset, there is a clear upward trend, showing the CO₂ values steadily increasing over time. The plot also shows a seasonal pattern that regularly repeats each year. The variance of the data looks fairly consistent throughout the time period, there’s no need to apply a logarithmic or other transformation to stabilize the data.

---

### **(b): Making the Series Stationary**

**Step 1: Remove Seasonality** 

```r
# Remove seasonality
co2_season_adj <- diff(co2, lag = 12)

# Plot
plot(co2_season_adj, main = "Lag-12 Differenced CO2 (Seasonality Removed)",
     ylab = "Change from 12 Months Ago", xlab = "Year")
```

![image.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/image%201.png)

A lag-12 difference was used to remove the seasonal pattern from the CO₂ data. This means we subtracted each value from the value 12 months earlier, which helps eliminate repeating yearly cycles. After applying this operation, the regular ups and downs that occurred each year are mostly gone, but an upward trend still appears. 

---

**Step 2: Remove Trend** 

```r
# Remove trend by differencing again
co2_stationary <- diff(co2_season_adj)

# Plot
plot(co2_stationary, main = "Differenced Lag-12 CO2 (Stationarity Attempt)",
     ylab = "Second Difference", xlab = "Year")
```

![image.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/image%202.png)

Next, a second differencing step was applied by subtracting each value from the one before it. This helps remove the overall upward trend that was still visible after the first step. The new plot appears to be stationary, the data no longer shows a clear trend or repeating seasonal patterns. The fluctuations now appear to be centered around a constant level, and their variance seems fairly consistent throughout the time series. 

---

## Task 2: Backshift Operator and Linear Filter

![Screenshot 2025-06-23 at 2.47.53 AM.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/Screenshot_2025-06-23_at_2.47.53_AM.png)

![Screenshot 2025-06-23 at 2.48.21 AM.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/Screenshot_2025-06-23_at_2.48.21_AM.png)

---

### (a) **Show that if** $\sum_{j=-2}^2 \psi_j = 1$ **and** $\sum_{j=-2}^2 j \psi_j = 0$**, then** $\psi(B)(a + bt) = a + bt$ **for all** $t \in \mathbb{Z}$**.**

The filter is:

$$
\psi(B)=\sum_{j=-2}^{2}\psi_j B^j
$$

When we apply this to $x_t = a + t b$, we get:

$$
\psi(B)(x_t)=\sum_{j=-2}^{2}\psi_j x_{t-j}
$$

But $x_{t-j} = a + (t - j)b = a + tb - jb$, so:

$$
\psi(B)(x_t) = \sum_{j=-2}^{2} \psi_j (a + tb - jb)
$$

Now break this sum into parts:

$$
\psi(B)(x_t) = \sum_{j=-2}^{2} \psi_j a + \sum_{j=-2}^{2} \psi_j tb - \sum_{j=-2}^{2} \psi_j jb
$$

Factor out the constants:

$$
\psi(B)(x_t) = a \sum_{j=-2}^{2} \psi_j + tb \sum_{j=-2}^{2} \psi_j - b \sum_{j=-2}^{2} j \psi_j
$$

Using the given conditions

1. $\sum_{j=-2}^{2} \psi_j = 1$
2. $\sum_{j=-2}^{2} j \psi_j = 0$

Plug these into our expression:

$$
\psi(B)(x_t) = a (1) + tb (1) - b (0) = a + tb
$$

**Final Result:**

$$
\psi(B)(a + tb) = a + tb
$$

So the filter returns the same linear trend we started with. This shows that $\psi(B)$ passes linear trends without distortion. So applying $\psi(B)$ to any linear trend just gives you that same trend back unchanged.

---

### (b) **Give an explicit example of a filter** $\psi(B)$ **satisfying both conditions from (a).**

Symmetry around zero naturally helps satisfy the second condition so we will choose a symmetric filter:

$$
\psi_{-2} = \psi_2 = \frac{1}{10}, \quad \psi_{-1} = \psi_1 = \frac{2}{10}, \quad \psi_0 = \frac{4}{10}
$$

Check the two conditions:

1. **Sum of coefficients**:
    
    $$
    \frac{1}{10} + \frac{2}{10} + \frac{4}{10} + \frac{2}{10} + \frac{1}{10} = \frac{10}{10} = 1
    $$
    
2. **Weighted sum of indices**:
    
    $$
    (-2)\cdot\frac{1}{10} + (-1)\cdot\frac{2}{10} + 0 + 1\cdot\frac{2}{10} + 2\cdot\frac{1}{10} = -\frac{2}{10} - \frac{2}{10} + \frac{2}{10} + \frac{2}{10} = 0
    $$
    

Therefore, the filter

$$
\psi(B) = \frac{1}{10}B^{-2} + \frac{2}{10}B^{-1} + \frac{4}{10} + \frac{2}{10}B + \frac{1}{10}B^2
$$

satisfies the desired properties.

---

### (c) **Give a filter** $\psi(B)$ **such that** $\sum \psi_j = 1$**, but the filter does not pass linear trends without distortion.**

Let:

$$
\psi_{-2} = 0.5,\quad \psi_{-1} = 0.2,\quad \psi_0 = 0.2,\quad \psi_1 = 0.05,\quad \psi_2 = 0.05
$$

1. **Sum of weights**:
    
    0.5 + 0.2 + 0.2 + 0.05 + 0.05 = 1
    
2. **Check** $\sum j\psi_j$:
    
    $$
    (-2)\cdot0.5 + (-1)\cdot0.2 + 0 + 1\cdot0.05 + 2\cdot0.05 = -1 - 0.2 + 0.05 + 0.1 = -1.05
    $$
    

Since this is not zero, the filter does not pass linear trends.

**Verification**: Let $x_t = a + bt = 1 + t$. Choose $t = 0$:

$$
x_{-2} = -1,\quad x_{-1} = 0,\quad x_0 = 1,\quad x_1 = 2,\quad x_2 = 3
$$

Now compute:

$$
\psi(B)x_0 = 0.5\cdot(-1) + 0.2\cdot 0 + 0.2\cdot1 + 0.05\cdot2 + 0.05\cdot3
$$

$$
= -0.5 + 0 + 0.2 + 0.1 + 0.15 = -0.05
$$

But $x_0 = 1$, so:

$$
\psi(B)x_0 = -0.05 \ne 1 = x_0
$$

Therefore, this filter **distorts linear trends**, as required.

---

## Task 3: Autocorrelation Function of an AR(2) Process

![Screenshot 2025-06-23 at 4.03.47 AM.png](Graded%20Assignment%201%2021a509a087318041a72aec793d719824/Screenshot_2025-06-23_at_4.03.47_AM.png)

The given AR(2) process is defined by:

$$
X_t - 0.8X_{t-1} + 0.2X_{t-2} = W_t, \quad W_t \sim \text{WN}(0, \sigma^2). 
$$

### (a) Factor AR Polynomial and argue causality

The autoregressive polynomial is:

$$
\phi(z) = 1 - 0.8z + 0.2z^2.
$$

To factor $\phi(z)$, we find the inverse roots using R:

```r
phi <- c(0.8, -0.2)
phi.zinv <- c(-phi[2], -phi[1], 1)
polyroot(phi.zinv)
```

Output:

```r
[1] 0.4+0.2i 0.4-0.2i
```

Thus, the roots are $2 + i$ and $2 - i$. The polynomial factors as:

$$
\phi(z) = (1 - (0.4 - 0.2i)z)(1 - (0.4 + 0.2i)z)
$$

**Verify causality**

For an AR(2) process to be **causal**, all roots of $\phi(z)$ must lie *outside* the unit circle ($|z| > 1$).

- Root $2 + i$:
$|2 + i| = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236 > 1$
- Root $2 - i$:
$|2 - i| = \sqrt{2^2 + (-1)^2} = \sqrt{5} \approx 2.236 > 1$

Both roots have magnitude $\sqrt{5} > 1$, so they lie outside the unit circle.

The process is causal because the roots of $\phi(z)$ are outside the unit circle. This ensures $X_t$ can be expressed solely in terms of past and present white noise terms $W_{t-j}$ $(j \geq 0)$, satisfying the causality condition.

---

### (b) Compute a formula for the autocorrelation function $\rho_X(h)$

**1: Write down the recurrence relation**

Multiply the AR equation by $X_{t-h}$, take expectations, and divide by $\gamma_X(0)$ to get the ACF difference equation:

$$
\rho(h) - 0.8 \rho(h-1) + 0.2 \rho(h-2) = 0, \quad h \geq 1
$$

**2: General solution form**

Since the roots of $\phi(z)$ are **complex and conjugate**, the solution has the form:

$$
\rho(h) = 2 \alpha \beta^h \cos(h \Phi + \Theta), \quad h \geq 0
$$

We’ll find the parameters $\alpha, \beta, \Phi, \Theta$ based on the complex root $r_1^{-1} = 0.5 + 0.5i$ and the corresponding coefficient $c_1$.

---

**3: Find $\rho(0)$ and $\rho(1)$**

We know:

$$
\rho(0) = 1
$$

To find $\rho(1)$, use the recurrence with evenness: $\rho(-1) = \rho(1)$

$$
0 = \rho(1) - 0.8 \cdot \rho(0) + 0.2 \cdot \rho(1) \Rightarrow (1 + 0.2) \rho(1) = 0.8 \Rightarrow \rho(1) = \frac{0.8}{1.2} = \frac{2}{3}
$$

---

**4: Solve for constants**

Using:

$$
\rho(0) = c_1 + c_2 = 1 \\ \rho(1) = c_1 r_1^{-1} + c_2 r_2^{-1} = \frac{2}{3}
$$

R code:

```r
phi <- c(0.8, -0.2)
phi.zinv <- c(-phi[2], -phi[1], 1)
r.inv <- polyroot(phi.zinv)  # roots of z^{-1} polynomial

A <- matrix(c(1, 1, r.inv[1], r.inv[2]), nrow=2, byrow=TRUE)
b <- c(1, phi[1]/(1 - phi[2]))
c <- solve(A, b)
c
```

**Output:**

```r
[1] 0.5-0.6666667i 0.5+0.6666667i
```

So:

- $c_1=0.5-0.6666667i$
- $c_2=0.5+0.6666667i$

Since they are complex conjugates, we rewrite the ACF formula as:

---

**Final Formula:**

We write $c_1 = \alpha e^{i\Theta},\ r_1^{-1} = \beta e^{i\Phi}$, so:

$$
\rho(h) = 2\alpha \beta^h \cos(h\Phi + \Theta)
$$

R code:

```r
alpha <- Mod(c[1])
Theta <- Arg(c[1])
beta <- Mod(r.inv[1])
Phi <- Arg(r.inv[1])
```

```r
rho.X <- numeric(10)
for(h in 1:10){
  rho.X[h] <- 2 * alpha * beta^h * cos(h * Phi + Theta)
}
rho.X <- Re(rho.X)
rho.X
```

**Output:**

```r
 [1]  0.6666666667  0.3333333333  0.1333333333  0.0400000000  0.0053333333 
		 -0.0037333333 -0.0040533333 -0.0024960000 -0.0011861333
[10] -0.0004497067
```

---

---

### (c) Compare with ARMAacf output

```r
ARMAacf(ar = phi, lag.max = 10)[2:11]
```

**Output:**

```r
            1             2             3             4             5             6          
 0.6666666667  0.3333333333  0.1333333333  0.0400000000  0.0053333333 -0.0037333333 
  7             8             9            10 
 -0.0040533333 -0.0024960000 -0.0011861333 -0.0004497067 
```

The outputs match **exactly**, confirming correctness.

---

**Final Answer Summary**

- **Causal:** Yes, roots have magnitude > 1
- **Formula:**

$$
\rho(h) = 2\alpha \beta^h \cos(h\Phi + \Theta)
\quad \text{where:} \quad \alpha \approx 0.501, \ \beta \approx 0.707, \ \Phi \approx 0.785, \ \Theta \approx -0.091
$$

- **ACF Values:**
    
    $$
    ⁍
    $$
