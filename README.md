# MPC COVID-19

Modelling and simulation of a COVID-19 compartmental system.

## Modelling

### Variables

| Variable       | Description                                 |
|:--------------:|:-------------------------------------------:|
| $P_k$          | Total individuals in age group $k$          |
| $S_k$          | Susceptible individuals from age group $k$  |
| $I_k$          | Infected individuals from age group $k$     |
| $R_k$          | Recovered individuals from age group $k$    |
| $D_k$          | Deceased individuals from age group $k$     |
| $U_k$          | Vaccination rate for age group $k$          |
| $\lambda_k$    | Infection rate for age group $k$            |
| $C_{j,k}$      | Contact rate between age groups $j$ and $k$ |
| $\gamma_{R,k}$ | Recovery rate for age group $k$             |
| $\gamma_{D,k}$ | Decease rate for age group $k$              |
| $n_a$          | Number of age groups                        |
| $\Delta_t$     | Sampling Period                             |

### Continous Model

$$
\left\{\begin{aligned}
\frac{d S_k(t)}{dt} &= -\lambda_k \cdot S_k(t) \cdot \sum_{j=1}^{n_a} C_{j,k} \cdot I_j(t) - U_k(t)
\\
\frac{d I_k(t)}{dt} &= \lambda_k \cdot S_k(t) \cdot \sum_{j=1}^{n_a} C_{k,j} \cdot I_j(t) - (\gamma_{R,k} + \gamma_{D,k}) \cdot I_k(t)
\\
\frac{d R_k(t)}{dt} &= \gamma_{R,k} \cdot I_k(t) + U_k(t)
\\
\frac{d D_k(t)}{dt} &= \gamma_{D,k} \cdot I_k(t)
\end{aligned}\right.
, k = 1, \dotsc, n_a
$$

### Discrete Model

$$
\left\{\begin{aligned}
S_k(n+1) &= S_k(n) + \Delta_t \cdot \left( -\lambda_k \cdot S_k(n) \cdot \sum_{j=1}^{n_a} C_{j,k} \cdot I_j(n) - U_k(n) \right)
\\
I_k(n+1) &= I_k(n) + \Delta_t \cdot \left( \lambda_k \cdot S_k(n) \cdot \sum_{j=1}^{n_a} C_{k,j} \cdot I_j(n) - (\gamma_{R,k} + \gamma_{D,k}) \cdot I_k(n) \right)
\\
R_k(n+1) &= R_k(n) + \Delta_t \cdot \left( \gamma_{R,k} \cdot I_k(n) + U_k(n) \right)
\\
D_k(n+1) &= D_k(n) + \Delta_t \cdot \left( \gamma_{D,k} \cdot I_k(n) \right)
\end{aligned}\right.
, k = 1, \dotsc, n_a
$$
Considering $\Delta_t = 1$
$$
\left\{\begin{aligned}
S_k(n+1) &= S_k(n) - \lambda_k \cdot S_k(n) \cdot \sum_{j=1}^{n_a} C_{j,k} \cdot I_j(n) - U_k(n)
\\
I_k(n+1) &= I_k(n) + \lambda_k \cdot S_k(n) \, \sum_{j=1}^{n_a} C_{k,j} \cdot I_j(n) - (\gamma_{R,k} + \gamma_{D,k}) \cdot I_k(n)
\\
R_k(n+1) &= R_k(n) + \gamma_{R,k} \cdot I_k(n) + U_k(n)
\\
D_k(n+1) &= D_k(n) + \gamma_{D,k} \cdot I_k(n)
\end{aligned}\right.
, k = 1, \dotsc, n_a
$$

## Pre-Simulation

### Definitions


```python
import numpy as np
from helpers import *

P, S_0, I_0, R_0, D_0, l, C, g_R, g_D, u_max = definitions()
y_0 = wrap(S_0, I_0, R_0, D_0)
n_a = len(P)

t_span = [0, 80]

print(f'Population Total: {sum(P):.0f}')
```

    Population Total: 3645243


## Continous Simulation


```python
def system_continuous(t, y, u, u_max, l, C, g_R, g_D):
  S, I, R, D = unwrap(y)
  dSdt = - l * S * (C @ I) - u(t, y, u_max)
  dIdt = l * S * (C @ I) - (g_R + g_D) * I
  dRdt = g_R * I + u(t, y, u_max)
  dDdt = g_D * I
  return wrap(dSdt, dIdt, dRdt, dDdt)
```

### No Vaccination


```python
from scipy.integrate import solve_ivp

def control(t, y, max):
  return np.zeros(n_a)

sol = solve_ivp(system_continuous, t_span, y_0, args=(control, u_max, l, C, g_R, g_D))
assert(sol.success)

t = sol.t
y = sol.y
S, I, R, D = unwrap(y)
u = recover_control(t, y, control, u_max)

plot(t, y, u)
print(f'Deceased Total: {sum(D[:,-1]):.0f}')
```


![png](README_files/README_11_0.png)


    Deceased Total: 46089


### Vaccination of Older Groups with Exclusivity


```python
from scipy.integrate import solve_ivp

def control(t, y, u_max):
  S, I, R, D = unwrap(y)
  u = np.zeros(n_a)
  for i in reversed(range(n_a)):
    if S[i] > 0:
      u[i] = u_max
      break
  return u

sol = solve_ivp(system_continuous, t_span, y_0, args=(control, u_max, l, C, g_R, g_D))
assert(sol.success)

t = sol.t
y = sol.y
S, I, R, D = unwrap(y)
u = recover_control(t, y, control, u_max)

plot(t, y, u)
print(f'Deceased Total: {sum(D[:,-1]):.0f}')
```


![png](README_files/README_13_0.png)


    Deceased Total: 7633


### Vaccination of Older Groups with Intersection


```python
from scipy.integrate import solve_ivp

def control(t, y, u_max):
  S, I, R, D = unwrap(y)
  u = np.zeros(n_a)
  remaining = u_max
  for i in reversed(range(n_a)):
    u[i] = min([S[i], remaining])
    remaining = remaining - u[i]
  return u

sol = solve_ivp(system_continuous, t_span, y_0, args=(control, u_max, l, C, g_R, g_D))
assert(sol.success)

t = sol.t
y = sol.y
S, I, R, D = unwrap(y)
u = recover_control(t, y, control, u_max)

plot(t, y, u)
print(f'Deceased Total: {sum(D[:,-1]):.0f}')
```


![png](README_files/README_15_0.png)


    Deceased Total: 7616


## Discrete Simulation


```python
def system_discrete(t, y, u, u_max, l, C, g_R, g_D):
  S, I, R, D = unwrap(y)
  S_ = S - l * S * (C @ I) - u(t, y, u_max)
  I_ = I + l * S * (C @ I) - (g_R + g_D) * I
  R_ = R + g_R * I + u(t, y, u_max)
  D_ = D + g_D * I
  return wrap(S_, I_, R_, D_)
```

### No Vaccination


```python
def control(t, y, u_max):
  return np.zeros(n_a)

t, y = solve_ivp_discrete(system_discrete, t_span, y_0, args=(control, u_max, l, C, g_R, g_D))

S, I, R, D = unwrap(y)
u = recover_control(t, y, control, u_max)

plot(t, y, u, discrete=True)
print(f'Deceased Total: {sum(D[:,-1]):.0f}')
```


![png](README_files/README_19_0.png)


    Deceased Total: 47115


### Vaccination of Older Groups


```python
def control(t, y, u_max):
  S, I, R, D = unwrap(y)
  u = np.zeros(n_a)
  remaining = u_max
  for i in reversed(range(n_a)):
    u[i] = min([S[i], remaining])
    remaining = remaining - u[i]
  return u

t, y = solve_ivp_discrete(system_discrete, t_span, y_0, args=(control, u_max, l, C, g_R, g_D))

S, I, R, D = unwrap(y)
u = recover_control(t, y, control, u_max)

plot(t, y, u, discrete=True)
print(f'Deceased Total: {sum(D[:,-1]):.0f}')
```


![png](README_files/README_21_0.png)


    Deceased Total: 7593


## MPC

### Definitions


```python
import numpy as np
import casadi
from helpers import *

P, S_0, I_0, R_0, D_0, l, C, g_R, g_D, u_max = definitions()
X_0 = wrap(S_0, I_0, R_0, D_0)
n_a = len(P)

S = casadi.MX.sym('S', n_a)
I = casadi.MX.sym('I', n_a)
R = casadi.MX.sym('R', n_a)
D = casadi.MX.sym('D', n_a)
U = casadi.MX.sym('U', n_a)
X = wrap(S, I, R, D)

def system_discrete(X, U):
  S, I, R, D = unwrap(X)
  S_ = S - l * S * (C @ I) - U
  I_ = I + l * S * (C @ I) - (g_R + g_D) * I
  R_ = R + g_R * I + U
  D_ = D + g_D * I
  return wrap(S_, I_, R_, D_)

f = casadi.Function('f',[X, U],[system_discrete(X, U)],['X', 'U'],['X+'])
print(f)
```

    f:(X[24],U[6])->(X+[24]) MXFunction


### Optimization


```python
opti = casadi.Opti()

N = 60 # prediction horizon

s = opti.variable(n_a, N + 1)
i = opti.variable(n_a, N + 1)
r = opti.variable(n_a, N + 1)
d = opti.variable(n_a, N + 1)
u = opti.variable(n_a, N)
x = wrap(s, i, r, d)

s_0 = opti.parameter(n_a,1)
i_0 = opti.parameter(n_a,1)
r_0 = opti.parameter(n_a,1)
d_0 = opti.parameter(n_a,1)
x_0 = wrap(s_0, i_0, r_0, d_0)

opti.minimize(0.1*casadi.sumsqr(i) + 0.9*casadi.sumsqr(d))

opti.subject_to(x[:,0] == x_0) # initial conditions
for k in range(N):
  opti.subject_to(x[:,k + 1] == f(x[:,k], u[:,k])) # dynamics
  opti.subject_to(u[:,k] <= s[:,k]) # dynamic control bound
opti.subject_to(opti.bounded(0,u,u_max)) # individual control bounds
opti.subject_to(opti.bounded(0,casadi.sum1(u),u_max)) # joint control bounds

opti.set_value(x_0, X_0)

opti.solver('ipopt', {}, {
  'print_level': 0,
})

sol = opti.solve()

t_grid = range(N + 1)
x_grid = sol.value(x)
u_grid = np.hstack((sol.value(u), np.nan*np.ones((n_a,1))))
plot(t_grid, x_grid, u_grid, discrete=True)

print(f'Deceased Total: {sum(x_grid[3*n_a:4*n_a,-1]):.0f}')
```

    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |        0 (       0)   4.43ms (  7.32us)       606
           nlp_g  | 210.00ms (346.53us) 157.29ms (259.55us)       606
      nlp_grad_f  |   8.00ms ( 13.96us)  11.82ms ( 20.63us)       573
      nlp_hess_l  |   1.47 s (  2.57ms)   1.43 s (  2.50ms)       570
       nlp_jac_g  |   1.68 s (  2.93ms)   1.80 s (  3.13ms)       574
           total  |  28.03 s ( 28.03 s)  28.03 s ( 28.03 s)         1



![png](README_files/README_26_1.png)


    Deceased Total: 3590


### MPC


```python
O = opti.to_function('M',[x_0],[u[:,0]],['x_0'],['u'])
print(O)

M = 1 # control horizon [?]

x_ = X_0
x_log = np.empty((0, x_0.shape[0]))
u_log = np.empty((0, u.shape[0]))
for i in range(N):
  if i % M == 0:
    u_ = np.array(O(x_))[:,0]

  x_log = np.vstack((x_log, x_))
  u_log = np.vstack((u_log, u_))

  x_ = np.array(f(x_,u_))[:,0]

t_log = range(N)
x_log = x_log.T
u_log = u_log.T
plot(t_log, x_log, u_log, discrete=True)

print(f'Deceased Total: {sum(x_log[3*n_a:4*n_a,-1]):.0f}')
```

    M:(x_0[24])->(u[6]) MXFunction
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  13.00ms ( 21.45us)   4.83ms (  7.97us)       606
           nlp_g  | 172.00ms (283.83us) 166.29ms (274.40us)       606
      nlp_grad_f  |        0 (       0)  13.06ms ( 22.79us)       573
      nlp_hess_l  |   1.50 s (  2.63ms)   1.47 s (  2.59ms)       570
       nlp_jac_g  |   1.85 s (  3.22ms)   1.89 s (  3.29ms)       574
           total  |  31.26 s ( 31.26 s)  31.25 s ( 31.25 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  13.00ms ( 11.29us)   9.46ms (  8.22us)      1151
           nlp_g  | 416.00ms (361.42us) 327.05ms (284.14us)      1151
        nlp_grad  |        0 (       0) 561.00us (561.00us)         1
      nlp_grad_f  |   9.00ms (  8.19us)  25.88ms ( 23.55us)      1099
      nlp_hess_l  |   2.92 s (  2.67ms)   2.89 s (  2.64ms)      1094
       nlp_jac_g  |   3.62 s (  3.28ms)   3.74 s (  3.39ms)      1102
           total  |  31.08 s ( 31.08 s)  31.08 s ( 31.08 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  13.00ms (  7.68us)  14.02ms (  8.28us)      1693
           nlp_g  | 606.00ms (357.94us) 481.18ms (284.22us)      1693
        nlp_grad  |        0 (       0)   1.11ms (557.50us)         2
      nlp_grad_f  |  18.00ms ( 11.09us)  38.43ms ( 23.68us)      1623
      nlp_hess_l  |   4.46 s (  2.75ms)   4.30 s (  2.65ms)      1621
       nlp_jac_g  |   5.47 s (  3.35ms)   5.58 s (  3.41ms)      1635
           total  |  31.14 s ( 31.14 s)  31.14 s ( 31.14 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  20.00ms (  8.90us)  18.82ms (  8.37us)      2247
           nlp_g  | 751.00ms (334.22us) 646.41ms (287.68us)      2247
        nlp_grad  |        0 (       0)   1.79ms (596.33us)         3
      nlp_grad_f  |  20.00ms (  9.33us)  50.96ms ( 23.77us)      2144
      nlp_hess_l  |   6.00 s (  2.80ms)   5.77 s (  2.69ms)      2145
       nlp_jac_g  |   7.44 s (  3.44ms)   7.49 s (  3.46ms)      2165
           total  |  31.68 s ( 31.68 s)  31.68 s ( 31.68 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  20.00ms (  7.37us)  22.68ms (  8.35us)      2715
           nlp_g  | 855.00ms (314.92us) 779.73ms (287.19us)      2715
        nlp_grad  |        0 (       0)   2.43ms (607.00us)         4
      nlp_grad_f  |  29.00ms ( 11.21us)  61.89ms ( 23.92us)      2588
      nlp_hess_l  |   7.12 s (  2.75ms)   6.96 s (  2.69ms)      2587
       nlp_jac_g  |   9.13 s (  3.50ms)   9.10 s (  3.48ms)      2611
           total  |  26.71 s ( 26.71 s)  26.71 s ( 26.71 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  20.00ms (  5.87us)  28.24ms (  8.28us)      3409
           nlp_g  |   1.02 s (298.33us) 977.86ms (286.85us)      3409
        nlp_grad  |        0 (       0)   3.12ms (624.80us)         5
      nlp_grad_f  |  36.00ms ( 11.48us)  75.15ms ( 23.97us)      3135
      nlp_hess_l  |   8.62 s (  2.74ms)   8.50 s (  2.71ms)      3141
       nlp_jac_g  |  11.16 s (  3.52ms)  11.13 s (  3.51ms)      3173
           total  |  35.61 s ( 35.61 s)  35.62 s ( 35.62 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  32.00ms (  8.11us)  33.01ms (  8.37us)      3945
           nlp_g  |   1.12 s (282.89us)   1.14 s (290.17us)      3945
        nlp_grad  |        0 (       0)   3.83ms (637.83us)         6
      nlp_grad_f  |  37.00ms ( 10.14us)  88.14ms ( 24.15us)      3650
      nlp_hess_l  |   9.96 s (  2.72ms)  10.02 s (  2.74ms)      3661
       nlp_jac_g  |  13.10 s (  3.54ms)  13.10 s (  3.54ms)      3699
           total  |  35.42 s ( 35.42 s)  35.42 s ( 35.42 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  33.00ms (  7.45us)  37.12ms (  8.38us)      4432
           nlp_g  |   1.19 s (267.60us)   1.28 s (289.08us)      4432
        nlp_grad  |        0 (       0)   4.34ms (619.57us)         7
      nlp_grad_f  |  37.00ms (  9.00us)  99.06ms ( 24.09us)      4112
      nlp_hess_l  |  11.15 s (  2.71ms)  11.23 s (  2.72ms)      4121
       nlp_jac_g  |  14.65 s (  3.52ms)  14.64 s (  3.52ms)      4161
           total  |  28.91 s ( 28.91 s)  28.91 s ( 28.91 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  35.00ms (  7.18us)  40.81ms (  8.37us)      4874
           nlp_g  |   1.24 s (254.62us)   1.41 s (288.97us)      4874
        nlp_grad  |        0 (       0)   5.00ms (625.50us)         8
      nlp_grad_f  |  37.00ms (  8.16us) 109.29ms ( 24.09us)      4537
      nlp_hess_l  |  12.33 s (  2.71ms)  12.35 s (  2.72ms)      4544
       nlp_jac_g  |  16.11 s (  3.51ms)  16.10 s (  3.51ms)      4586
           total  |  26.90 s ( 26.90 s)  26.89 s ( 26.89 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  36.00ms (  6.84us)  44.15ms (  8.39us)      5265
           nlp_g  |   1.38 s (261.73us)   1.52 s (289.35us)      5265
        nlp_grad  |        0 (       0)   5.79ms (643.89us)         9
      nlp_grad_f  |  51.00ms ( 10.35us) 118.93ms ( 24.13us)      4929
      nlp_hess_l  |  13.38 s (  2.71ms)  13.49 s (  2.74ms)      4934
       nlp_jac_g  |  17.71 s (  3.56ms)  17.49 s (  3.51ms)      4978
           total  |  24.40 s ( 24.40 s)  24.40 s ( 24.40 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  40.00ms (  7.08us)  47.60ms (  8.43us)      5648
           nlp_g  |   1.48 s (261.86us)   1.63 s (289.35us)      5648
        nlp_grad  |   1.00ms (100.00us)   6.52ms (652.00us)        10
      nlp_grad_f  |  55.00ms ( 10.36us) 127.85ms ( 24.09us)      5307
      nlp_hess_l  |  14.32 s (  2.70ms)  14.50 s (  2.73ms)      5310
       nlp_jac_g  |  19.14 s (  3.57ms)  18.80 s (  3.51ms)      5356
           total  |  23.71 s ( 23.71 s)  23.71 s ( 23.71 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  56.00ms (  9.29us)  50.04ms (  8.30us)      6027
           nlp_g  |   1.58 s (261.99us)   1.72 s (284.61us)      6027
        nlp_grad  |   1.00ms ( 90.91us)   7.17ms (652.18us)        11
      nlp_grad_f  |  56.00ms (  9.86us) 134.72ms ( 23.71us)      5682
      nlp_hess_l  |  15.13 s (  2.66ms)  15.26 s (  2.68ms)      5683
       nlp_jac_g  |  20.07 s (  3.50ms)  19.78 s (  3.45ms)      5731
           total  |  17.28 s ( 17.28 s)  17.28 s ( 17.28 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  56.00ms (  8.76us)  51.95ms (  8.13us)      6390
           nlp_g  |   1.60 s (250.08us)   1.77 s (277.58us)      6390
        nlp_grad  |   1.00ms ( 83.33us)   7.83ms (652.92us)        12
      nlp_grad_f  |  71.00ms ( 11.77us) 139.53ms ( 23.12us)      6034
      nlp_hess_l  |  15.60 s (  2.59ms)  15.76 s (  2.61ms)      6033
       nlp_jac_g  |  20.67 s (  3.40ms)  20.43 s (  3.36ms)      6083
           total  |  11.03 s ( 11.03 s)  11.02 s ( 11.02 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  58.00ms (  8.61us)  53.71ms (  7.97us)      6739
           nlp_g  |   1.62 s (241.13us)   1.83 s (271.65us)      6739
        nlp_grad  |   1.00ms ( 76.92us)   8.49ms (653.00us)        13
      nlp_grad_f  |  75.00ms ( 11.75us) 144.15ms ( 22.58us)      6384
      nlp_hess_l  |  16.19 s (  2.54ms)  16.24 s (  2.55ms)      6381
       nlp_jac_g  |  21.36 s (  3.32ms)  21.08 s (  3.28ms)      6433
           total  |  11.89 s ( 11.89 s)  11.89 s ( 11.89 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  58.00ms (  8.19us)  56.23ms (  7.94us)      7086
           nlp_g  |   1.66 s (234.26us)   1.92 s (270.35us)      7086
        nlp_grad  |   1.00ms ( 71.43us)   8.84ms (631.57us)        14
      nlp_grad_f  |  75.00ms ( 11.17us) 151.00ms ( 22.50us)      6712
      nlp_hess_l  |  16.96 s (  2.53ms)  16.98 s (  2.53ms)      6705
       nlp_jac_g  |  22.36 s (  3.31ms)  22.02 s (  3.26ms)      6763
           total  |  18.32 s ( 18.32 s)  18.32 s ( 18.32 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  58.00ms (  7.84us)  59.72ms (  8.08us)      7395
           nlp_g  |   1.75 s (237.05us)   2.02 s (272.89us)      7395
        nlp_grad  |   1.00ms ( 66.67us)   9.53ms (635.33us)        15
      nlp_grad_f  |  98.00ms ( 13.99us) 159.34ms ( 22.74us)      7007
      nlp_hess_l  |  18.00 s (  2.57ms)  17.89 s (  2.56ms)      6998
       nlp_jac_g  |  23.41 s (  3.32ms)  23.19 s (  3.29ms)      7058
           total  |  20.74 s ( 20.74 s)  20.73 s ( 20.73 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  8.56us)  64.31ms (  8.34us)      7714
           nlp_g  |   1.93 s (250.58us)   2.13 s (276.35us)      7714
        nlp_grad  |   1.00ms ( 62.50us)  10.43ms (651.81us)        16
      nlp_grad_f  | 116.00ms ( 15.84us) 169.14ms ( 23.09us)      7325
      nlp_hess_l  |  19.06 s (  2.61ms)  18.93 s (  2.59ms)      7314
       nlp_jac_g  |  24.72 s (  3.35ms)  24.56 s (  3.33ms)      7376
           total  |  24.06 s ( 24.06 s)  24.06 s ( 24.06 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  8.26us)  67.10ms (  8.39us)      7995
           nlp_g  |   2.00 s (250.66us)   2.23 s (278.59us)      7995
        nlp_grad  |   1.00ms ( 58.82us)  11.08ms (651.82us)        17
      nlp_grad_f  | 118.00ms ( 15.51us) 176.68ms ( 23.23us)      7606
      nlp_hess_l  |  19.93 s (  2.62ms)  19.80 s (  2.61ms)      7593
       nlp_jac_g  |  25.82 s (  3.37ms)  25.70 s (  3.36ms)      7657
           total  |  18.85 s ( 18.85 s)  18.85 s ( 18.85 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  7.96us)  69.70ms (  8.41us)      8291
           nlp_g  |   2.09 s (252.08us)   2.32 s (279.61us)      8291
        nlp_grad  |   1.00ms ( 55.56us)  11.60ms (644.56us)        18
      nlp_grad_f  | 141.00ms ( 17.84us) 184.68ms ( 23.37us)      7903
      nlp_hess_l  |  20.80 s (  2.64ms)  20.64 s (  2.62ms)      7888
       nlp_jac_g  |  26.90 s (  3.38ms)  26.79 s (  3.37ms)      7954
           total  |  18.38 s ( 18.38 s)  18.38 s ( 18.38 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  7.71us)  70.76ms (  8.27us)      8555
           nlp_g  |   2.11 s (246.99us)   2.35 s (274.71us)      8555
        nlp_grad  |   1.00ms ( 52.63us)  11.89ms (625.63us)        19
      nlp_grad_f  | 141.00ms ( 17.26us) 187.29ms ( 22.93us)      8168
      nlp_hess_l  |  21.09 s (  2.59ms)  20.92 s (  2.57ms)      8151
       nlp_jac_g  |  27.30 s (  3.32ms)  27.14 s (  3.30ms)      8219
           total  |   6.89 s (  6.89 s)   6.89 s (  6.89 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  7.45us)  71.99ms (  8.12us)      8862
           nlp_g  |   2.14 s (241.48us)   2.39 s (269.15us)      8862
        nlp_grad  |   1.00ms ( 50.00us)  12.07ms (603.65us)        20
      nlp_grad_f  | 141.00ms ( 16.67us) 190.23ms ( 22.50us)      8456
      nlp_hess_l  |  21.47 s (  2.54ms)  21.21 s (  2.51ms)      8437
       nlp_jac_g  |  27.64 s (  3.25ms)  27.52 s (  3.24ms)      8507
           total  |   8.23 s (  8.23 s)   8.23 s (  8.23 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  7.06us)  73.89ms (  7.91us)      9342
           nlp_g  |   2.19 s (234.32us)   2.44 s (261.41us)      9342
        nlp_grad  |   1.00ms ( 47.62us)  12.26ms (583.62us)        21
      nlp_grad_f  | 153.00ms ( 17.26us) 194.54ms ( 21.94us)      8866
      nlp_hess_l  |  21.91 s (  2.47ms)  21.66 s (  2.44ms)      8864
       nlp_jac_g  |  28.21 s (  3.16ms)  28.10 s (  3.14ms)      8938
           total  |  15.13 s ( 15.13 s)  15.14 s ( 15.14 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  6.76us)  75.44ms (  7.73us)      9759
           nlp_g  |   2.25 s (231.07us)   2.49 s (255.27us)      9759
        nlp_grad  |   1.00ms ( 45.45us)  12.47ms (566.77us)        22
      nlp_grad_f  | 153.00ms ( 16.63us) 198.10ms ( 21.54us)      9199
      nlp_hess_l  |  22.27 s (  2.42ms)  22.02 s (  2.39ms)      9194
       nlp_jac_g  |  28.62 s (  3.09ms)  28.58 s (  3.08ms)      9274
           total  |  10.20 s ( 10.20 s)  10.20 s ( 10.20 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  6.56us)  76.65ms (  7.62us)     10060
           nlp_g  |   2.26 s (224.95us)   2.53 s (251.35us)     10060
        nlp_grad  |   1.00ms ( 43.48us)  12.65ms (549.83us)        23
      nlp_grad_f  | 153.00ms ( 16.13us) 201.16ms ( 21.21us)      9484
      nlp_hess_l  |  22.61 s (  2.39ms)  22.32 s (  2.35ms)      9477
       nlp_jac_g  |  29.04 s (  3.04ms)  28.98 s (  3.03ms)      9559
           total  |   8.45 s (  8.45 s)   8.45 s (  8.45 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  6.37us)  77.91ms (  7.52us)     10357
           nlp_g  |   2.29 s (220.91us)   2.56 s (247.56us)     10357
        nlp_grad  |   1.00ms ( 41.67us)  12.92ms (538.37us)        24
      nlp_grad_f  | 179.00ms ( 18.32us) 204.18ms ( 20.90us)      9770
      nlp_hess_l  |  22.92 s (  2.35ms)  22.61 s (  2.32ms)      9759
       nlp_jac_g  |  29.36 s (  2.98ms)  29.37 s (  2.98ms)      9847
           total  |   8.95 s (  8.95 s)   8.96 s (  8.96 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  6.17us)  79.26ms (  7.41us)     10692
           nlp_g  |   2.31 s (216.14us)   2.61 s (243.65us)     10692
        nlp_grad  |   1.00ms ( 40.00us)  13.12ms (524.68us)        25
      nlp_grad_f  | 191.00ms ( 18.93us) 207.55ms ( 20.57us)     10088
      nlp_hess_l  |  23.21 s (  2.30ms)  22.94 s (  2.28ms)     10075
       nlp_jac_g  |  29.79 s (  2.93ms)  29.80 s (  2.93ms)     10165
           total  |  12.02 s ( 12.02 s)  12.01 s ( 12.01 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  6.03us)  80.27ms (  7.33us)     10943
           nlp_g  |   2.33 s (212.74us)   2.64 s (240.80us)     10943
        nlp_grad  |   1.00ms ( 38.46us)  13.31ms (511.77us)        26
      nlp_grad_f  | 191.00ms ( 18.50us) 209.99ms ( 20.34us)     10325
      nlp_hess_l  |  23.43 s (  2.27ms)  23.19 s (  2.25ms)     10310
       nlp_jac_g  |  30.15 s (  2.90ms)  30.13 s (  2.90ms)     10402
           total  |   7.73 s (  7.73 s)   7.73 s (  7.73 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  5.89us)  81.24ms (  7.25us)     11205
           nlp_g  |   2.37 s (211.51us)   2.67 s (237.89us)     11205
        nlp_grad  |   1.00ms ( 37.04us)  13.53ms (501.26us)        27
      nlp_grad_f  | 203.00ms ( 19.21us) 212.51ms ( 20.11us)     10567
      nlp_hess_l  |  23.60 s (  2.24ms)  23.43 s (  2.22ms)     10550
       nlp_jac_g  |  30.49 s (  2.86ms)  30.45 s (  2.86ms)     10644
           total  |   7.98 s (  7.98 s)   7.98 s (  7.98 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  5.75us)  82.30ms (  7.17us)     11475
           nlp_g  |   2.42 s (211.07us)   2.70 s (235.03us)     11475
        nlp_grad  |   1.00ms ( 35.71us)  13.92ms (497.11us)        28
      nlp_grad_f  | 203.00ms ( 18.73us) 215.10ms ( 19.85us)     10836
      nlp_hess_l  |  23.80 s (  2.20ms)  23.70 s (  2.19ms)     10817
       nlp_jac_g  |  30.80 s (  2.82ms)  30.80 s (  2.82ms)     10913
           total  |  10.43 s ( 10.43 s)  10.43 s ( 10.43 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  5.29us)  85.99ms (  6.89us)     12476
           nlp_g  |   2.51 s (201.59us)   2.81 s (225.08us)     12476
        nlp_grad  |   1.00ms ( 34.48us)  14.50ms (499.93us)        29
      nlp_grad_f  | 216.00ms ( 18.61us) 222.74ms ( 19.19us)     11607
      nlp_hess_l  |  24.67 s (  2.13ms)  24.46 s (  2.11ms)     11586
       nlp_jac_g  |  31.81 s (  2.72ms)  31.79 s (  2.72ms)     11684
           total  |  45.23 s ( 45.23 s)  45.24 s ( 45.24 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  5.17us)  87.07ms (  6.83us)     12755
           nlp_g  |   2.55 s (200.08us)   2.84 s (222.71us)     12755
        nlp_grad  |   1.00ms ( 33.33us)  14.68ms (489.30us)        30
      nlp_grad_f  | 216.00ms ( 18.19us) 225.39ms ( 18.98us)     11875
      nlp_hess_l  |  24.91 s (  2.10ms)  24.74 s (  2.09ms)     11852
       nlp_jac_g  |  32.10 s (  2.69ms)  32.15 s (  2.69ms)     11952
           total  |  10.92 s ( 10.92 s)  10.93 s ( 10.93 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  66.00ms (  5.08us)  87.94ms (  6.77us)     12996
           nlp_g  |   2.55 s (196.45us)   2.87 s (220.64us)     12996
        nlp_grad  |   1.00ms ( 32.26us)  14.85ms (479.13us)        31
      nlp_grad_f  | 216.00ms ( 17.91us) 227.22ms ( 18.84us)     12062
      nlp_hess_l  |  25.20 s (  2.09ms)  24.97 s (  2.07ms)     12085
       nlp_jac_g  |  32.41 s (  2.66ms)  32.45 s (  2.66ms)     12189
           total  |   7.37 s (  7.37 s)   7.37 s (  7.37 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  67.00ms (  5.02us)  89.65ms (  6.72us)     13341
           nlp_g  |   2.60 s (194.66us)   2.92 s (219.02us)     13341
        nlp_grad  |   1.00ms ( 31.25us)  15.08ms (471.12us)        32
      nlp_grad_f  | 216.00ms ( 17.46us) 231.36ms ( 18.70us)     12369
      nlp_hess_l  |  25.65 s (  2.07ms)  25.40 s (  2.05ms)     12390
       nlp_jac_g  |  32.95 s (  2.64ms)  33.05 s (  2.64ms)     12496
           total  |  20.88 s ( 20.88 s)  20.87 s ( 20.87 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  68.00ms (  4.97us)  92.82ms (  6.78us)     13694
           nlp_g  |   2.74 s (200.31us)   3.04 s (221.80us)     13694
        nlp_grad  |   1.00ms ( 30.30us)  15.73ms (476.67us)        33
      nlp_grad_f  | 216.00ms ( 17.00us) 240.02ms ( 18.89us)     12706
      nlp_hess_l  |  26.65 s (  2.09ms)  26.41 s (  2.08ms)     12724
       nlp_jac_g  |  34.19 s (  2.66ms)  34.37 s (  2.68ms)     12834
           total  |  38.77 s ( 38.77 s)  38.77 s ( 38.77 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  68.00ms (  4.88us)  94.98ms (  6.82us)     13924
           nlp_g  |   2.86 s (205.62us)   3.11 s (223.59us)     13924
        nlp_grad  |   1.00ms ( 29.41us)  16.39ms (482.12us)        34
      nlp_grad_f  | 221.00ms ( 17.08us) 246.03ms ( 19.02us)     12936
      nlp_hess_l  |  27.32 s (  2.11ms)  27.11 s (  2.09ms)     12952
       nlp_jac_g  |  35.14 s (  2.69ms)  35.28 s (  2.70ms)     13064
           total  |  25.50 s ( 25.50 s)  25.51 s ( 25.51 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  68.00ms (  4.80us)  97.15ms (  6.86us)     14159
           nlp_g  |   2.95 s (208.35us)   3.19 s (225.28us)     14159
        nlp_grad  |   1.00ms ( 28.57us)  17.33ms (495.11us)        35
      nlp_grad_f  | 236.00ms ( 17.92us) 251.95ms ( 19.14us)     13166
      nlp_hess_l  |  28.05 s (  2.13ms)  27.80 s (  2.11ms)     13179
       nlp_jac_g  |  36.01 s (  2.71ms)  36.21 s (  2.72ms)     13295
           total  |  24.49 s ( 24.49 s)  24.49 s ( 24.49 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  68.00ms (  4.73us)  99.16ms (  6.90us)     14375
           nlp_g  |   2.99 s (207.72us)   3.26 s (226.82us)     14375
        nlp_grad  |   1.00ms ( 27.78us)  18.01ms (500.42us)        36
      nlp_grad_f  | 243.00ms ( 18.16us) 257.43ms ( 19.24us)     13382
      nlp_hess_l  |  28.81 s (  2.15ms)  28.46 s (  2.12ms)     13393
       nlp_jac_g  |  36.85 s (  2.73ms)  37.07 s (  2.74ms)     13511
           total  |  25.89 s ( 25.89 s)  25.88 s ( 25.88 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  68.00ms (  4.64us) 101.64ms (  6.94us)     14652
           nlp_g  |   3.07 s (209.60us)   3.35 s (228.50us)     14652
        nlp_grad  |   1.00ms ( 27.03us)  18.55ms (501.49us)        37
      nlp_grad_f  | 243.00ms ( 17.83us) 263.88ms ( 19.36us)     13631
      nlp_hess_l  |  29.50 s (  2.16ms)  29.21 s (  2.14ms)     13640
       nlp_jac_g  |  37.90 s (  2.75ms)  38.05 s (  2.76ms)     13760
           total  |  32.42 s ( 32.42 s)  32.42 s ( 32.42 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  85.00ms (  5.74us) 118.19ms (  7.98us)     14809
           nlp_g  |   3.10 s (209.53us)   3.40 s (229.53us)     14809
        nlp_grad  |   1.00ms ( 26.32us)  19.93ms (524.58us)        38
      nlp_grad_f  | 243.00ms ( 17.63us) 267.91ms ( 19.44us)     13783
      nlp_hess_l  |  30.00 s (  2.18ms)  29.67 s (  2.15ms)     13790
       nlp_jac_g  |  38.47 s (  2.77ms)  38.64 s (  2.78ms)     13912
           total  |  14.71 s ( 14.71 s)  14.71 s ( 14.71 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  88.00ms (  5.88us) 119.54ms (  7.99us)     14967
           nlp_g  |   3.15 s (210.66us)   3.44 s (230.14us)     14967
        nlp_grad  |   1.00ms ( 25.64us)  20.60ms (528.15us)        39
      nlp_grad_f  | 256.00ms ( 18.36us) 271.55ms ( 19.48us)     13940
      nlp_hess_l  |  30.41 s (  2.18ms)  30.08 s (  2.16ms)     13945
       nlp_jac_g  |  38.96 s (  2.77ms)  39.18 s (  2.79ms)     14069
           total  |  12.29 s ( 12.29 s)  12.29 s ( 12.29 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  88.00ms (  5.81us) 120.52ms (  7.96us)     15134
           nlp_g  |   3.20 s (211.18us)   3.48 s (229.68us)     15134
        nlp_grad  |   1.00ms ( 25.00us)  20.79ms (519.72us)        40
      nlp_grad_f  | 272.00ms ( 19.28us) 274.07ms ( 19.43us)     14106
      nlp_hess_l  |  30.65 s (  2.17ms)  30.36 s (  2.15ms)     14109
       nlp_jac_g  |  39.39 s (  2.77ms)  39.56 s (  2.78ms)     14235
           total  |  13.46 s ( 13.46 s)  13.47 s ( 13.47 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  88.00ms (  5.68us) 123.54ms (  7.97us)     15501
           nlp_g  |   3.24 s (208.89us)   3.58 s (230.69us)     15501
        nlp_grad  |   1.00ms ( 24.39us)  21.46ms (523.44us)        41
      nlp_grad_f  | 287.00ms ( 19.86us) 281.83ms ( 19.50us)     14453
      nlp_hess_l  |  31.67 s (  2.19ms)  31.24 s (  2.16ms)     14454
       nlp_jac_g  |  40.49 s (  2.78ms)  40.70 s (  2.79ms)     14582
           total  |  43.11 s ( 43.11 s)  43.12 s ( 43.12 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  88.00ms (  5.61us) 124.29ms (  7.92us)     15683
           nlp_g  |   3.24 s (206.47us)   3.60 s (229.60us)     15683
        nlp_grad  |   1.00ms ( 23.81us)  21.67ms (515.90us)        42
      nlp_grad_f  | 287.00ms ( 19.64us) 283.53ms ( 19.40us)     14613
      nlp_hess_l  |  31.84 s (  2.18ms)  31.42 s (  2.15ms)     14612
       nlp_jac_g  |  40.71 s (  2.76ms)  40.94 s (  2.78ms)     14742
           total  |   7.33 s (  7.33 s)   7.33 s (  7.33 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  95.00ms (  6.00us) 125.13ms (  7.90us)     15842
           nlp_g  |   3.28 s (207.30us)   3.62 s (228.78us)     15842
        nlp_grad  |   1.00ms ( 23.26us)  21.94ms (510.16us)        43
      nlp_grad_f  | 287.00ms ( 19.43us) 285.52ms ( 19.33us)     14773
      nlp_hess_l  |  32.05 s (  2.17ms)  31.63 s (  2.14ms)     14770
       nlp_jac_g  |  40.98 s (  2.75ms)  41.20 s (  2.76ms)     14902
           total  |   8.33 s (  8.33 s)   8.34 s (  8.34 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  95.00ms (  5.94us) 125.71ms (  7.86us)     15995
           nlp_g  |   3.31 s (206.88us)   3.64 s (227.64us)     15995
        nlp_grad  |   1.00ms ( 22.73us)  22.17ms (503.89us)        44
      nlp_grad_f  | 287.00ms ( 19.23us) 286.91ms ( 19.22us)     14926
      nlp_hess_l  |  32.22 s (  2.16ms)  31.77 s (  2.13ms)     14921
       nlp_jac_g  |  41.15 s (  2.73ms)  41.39 s (  2.75ms)     15055
           total  |   6.48 s (  6.48 s)   6.48 s (  6.48 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  97.00ms (  5.96us) 126.90ms (  7.80us)     16266
           nlp_g  |   3.36 s (206.81us)   3.68 s (226.07us)     16266
        nlp_grad  |   1.00ms ( 22.22us)  22.48ms (499.51us)        45
      nlp_grad_f  | 287.00ms ( 18.94us) 289.37ms ( 19.10us)     15150
      nlp_hess_l  |  32.50 s (  2.15ms)  32.05 s (  2.12ms)     15143
       nlp_jac_g  |  41.50 s (  2.72ms)  41.75 s (  2.73ms)     15281
           total  |  11.11 s ( 11.11 s)  11.11 s ( 11.11 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  97.00ms (  5.91us) 127.51ms (  7.76us)     16422
           nlp_g  |   3.38 s (205.64us)   3.70 s (225.03us)     16422
        nlp_grad  |   1.00ms ( 21.74us)  22.67ms (492.89us)        46
      nlp_grad_f  | 287.00ms ( 18.77us) 290.74ms ( 19.01us)     15293
      nlp_hess_l  |  32.65 s (  2.14ms)  32.20 s (  2.11ms)     15284
       nlp_jac_g  |  41.68 s (  2.70ms)  41.94 s (  2.72ms)     15424
           total  |   6.29 s (  6.29 s)   6.29 s (  6.29 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  97.00ms (  5.87us) 127.91ms (  7.74us)     16525
           nlp_g  |   3.38 s (204.54us)   3.71 s (224.33us)     16525
        nlp_grad  |   1.00ms ( 21.28us)  22.87ms (486.70us)        47
      nlp_grad_f  | 287.00ms ( 18.64us) 291.65ms ( 18.95us)     15394
      nlp_hess_l  |  32.78 s (  2.13ms)  32.30 s (  2.10ms)     15383
       nlp_jac_g  |  41.83 s (  2.69ms)  42.07 s (  2.71ms)     15525
           total  |   3.85 s (  3.85 s)   3.85 s (  3.85 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  97.00ms (  5.77us) 130.31ms (  7.75us)     16811
           nlp_g  |   3.43 s (204.33us)   3.79 s (225.23us)     16811
        nlp_grad  |   1.00ms ( 20.83us)  23.39ms (487.35us)        48
      nlp_grad_f  | 287.00ms ( 18.42us) 296.02ms ( 18.99us)     15584
      nlp_hess_l  |  33.22 s (  2.13ms)  32.84 s (  2.11ms)     15590
       nlp_jac_g  |  42.49 s (  2.70ms)  42.79 s (  2.72ms)     15739
           total  |  22.73 s ( 22.73 s)  22.72 s ( 22.72 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  98.00ms (  5.80us) 131.23ms (  7.76us)     16911
           nlp_g  |   3.49 s (206.32us)   3.81 s (225.55us)     16911
        nlp_grad  |   1.00ms ( 20.41us)  23.92ms (488.18us)        49
      nlp_grad_f  | 287.00ms ( 18.30us) 298.18ms ( 19.01us)     15684
      nlp_hess_l  |  33.46 s (  2.13ms)  33.09 s (  2.11ms)     15688
       nlp_jac_g  |  42.79 s (  2.70ms)  43.12 s (  2.72ms)     15839
           total  |   9.02 s (  9.02 s)   9.03 s (  9.03 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  98.00ms (  5.75us) 132.47ms (  7.78us)     17036
           nlp_g  |   3.56 s (208.73us)   3.85 s (226.01us)     17036
        nlp_grad  |   1.00ms ( 20.00us)  24.47ms (489.44us)        50
      nlp_grad_f  | 287.00ms ( 18.16us) 301.17ms ( 19.05us)     15807
      nlp_hess_l  |  33.79 s (  2.14ms)  33.41 s (  2.11ms)     15809
       nlp_jac_g  |  43.35 s (  2.72ms)  43.54 s (  2.73ms)     15962
           total  |  13.49 s ( 13.49 s)  13.49 s ( 13.49 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  98.00ms (  5.72us) 133.31ms (  7.78us)     17133
           nlp_g  |   3.56 s (207.55us)   3.88 s (226.36us)     17133
        nlp_grad  |   1.00ms ( 19.61us)  25.19ms (493.84us)        51
      nlp_grad_f  | 287.00ms ( 18.06us) 303.14ms ( 19.07us)     15895
      nlp_hess_l  |  33.96 s (  2.14ms)  33.64 s (  2.12ms)     15895
       nlp_jac_g  |  43.75 s (  2.73ms)  43.84 s (  2.73ms)     16050
           total  |   7.96 s (  7.96 s)   7.96 s (  7.96 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  98.00ms (  5.66us) 134.74ms (  7.79us)     17308
           nlp_g  |   3.60 s (208.17us)   3.93 s (226.80us)     17308
        nlp_grad  |   1.00ms ( 19.23us)  25.86ms (497.25us)        52
      nlp_grad_f  | 287.00ms ( 17.91us) 306.29ms ( 19.11us)     16027
      nlp_hess_l  |  34.35 s (  2.14ms)  33.98 s (  2.12ms)     16025
       nlp_jac_g  |  44.33 s (  2.74ms)  44.29 s (  2.74ms)     16182
           total  |  15.62 s ( 15.62 s)  15.64 s ( 15.64 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.54us) 135.65ms (  7.78us)     17427
           nlp_g  |   3.62 s (207.67us)   3.96 s (227.05us)     17427
        nlp_grad  |   1.00ms ( 18.87us)  26.41ms (498.32us)        53
      nlp_grad_f  | 287.00ms ( 17.82us) 308.19ms ( 19.13us)     16110
      nlp_hess_l  |  34.60 s (  2.15ms)  34.20 s (  2.12ms)     16106
       nlp_jac_g  |  44.61 s (  2.74ms)  44.57 s (  2.74ms)     16265
           total  |   7.45 s (  7.45 s)   7.47 s (  7.47 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.51us) 136.42ms (  7.79us)     17513
           nlp_g  |   3.67 s (209.44us)   3.98 s (227.34us)     17513
        nlp_grad  |   1.00ms ( 18.52us)  26.96ms (499.26us)        54
      nlp_grad_f  | 287.00ms ( 17.72us) 310.17ms ( 19.15us)     16193
      nlp_hess_l  |  34.75 s (  2.15ms)  34.42 s (  2.13ms)     16187
       nlp_jac_g  |  44.88 s (  2.75ms)  44.86 s (  2.74ms)     16348
           total  |   7.97 s (  7.97 s)   7.98 s (  7.98 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.40us) 138.64ms (  7.78us)     17814
           nlp_g  |   3.74 s (210.23us)   4.06 s (228.06us)     17814
        nlp_grad  |   1.00ms ( 18.18us)  27.66ms (502.87us)        55
      nlp_grad_f  | 288.00ms ( 17.59us) 314.16ms ( 19.19us)     16373
      nlp_hess_l  |  35.17 s (  2.15ms)  34.88 s (  2.13ms)     16365
       nlp_jac_g  |  45.44 s (  2.75ms)  45.46 s (  2.75ms)     16528
           total  |  26.33 s ( 26.33 s)  26.34 s ( 26.34 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.37us) 139.36ms (  7.78us)     17908
           nlp_g  |   3.76 s (210.02us)   4.09 s (228.21us)     17908
        nlp_grad  |   1.00ms ( 17.86us)  28.21ms (503.82us)        56
      nlp_grad_f  | 288.00ms ( 17.52us) 315.81ms ( 19.21us)     16442
      nlp_hess_l  |  35.39 s (  2.15ms)  35.05 s (  2.13ms)     16432
       nlp_jac_g  |  45.64 s (  2.75ms)  45.69 s (  2.75ms)     16597
           total  |   6.36 s (  6.36 s)   6.36 s (  6.36 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.29us) 140.93ms (  7.77us)     18129
           nlp_g  |   3.82 s (210.88us)   4.14 s (228.56us)     18129
        nlp_grad  |   1.00ms ( 17.54us)  28.90ms (507.09us)        57
      nlp_grad_f  | 288.00ms ( 17.40us) 318.44ms ( 19.24us)     16547
      nlp_hess_l  |  35.64 s (  2.16ms)  35.33 s (  2.14ms)     16535
       nlp_jac_g  |  45.96 s (  2.75ms)  46.05 s (  2.76ms)     16702
           total  |  13.33 s ( 13.33 s)  13.33 s ( 13.33 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 114.00ms (  6.20us) 142.03ms (  7.73us)     18385
           nlp_g  |   3.84 s (209.08us)   4.18 s (227.49us)     18385
        nlp_grad  |   1.00ms ( 17.24us)  30.17ms (520.26us)        58
      nlp_grad_f  | 288.00ms ( 17.28us) 320.95ms ( 19.26us)     16664
      nlp_hess_l  |  35.96 s (  2.16ms)  35.56 s (  2.14ms)     16652
       nlp_jac_g  |  46.28 s (  2.75ms)  46.37 s (  2.76ms)     16825
           total  |   8.83 s (  8.83 s)   8.83 s (  8.83 s)         1
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  | 125.00ms (  6.67us) 144.07ms (  7.68us)     18754
           nlp_g  |   3.90 s (207.96us)   4.26 s (227.09us)     18754
        nlp_grad  |   1.00ms ( 16.95us)  30.59ms (518.42us)        59
      nlp_grad_f  | 288.00ms ( 17.11us) 323.80ms ( 19.24us)     16829
      nlp_hess_l  |  36.26 s (  2.16ms)  35.88 s (  2.13ms)     16815
       nlp_jac_g  |  46.70 s (  2.75ms)  46.79 s (  2.75ms)     16990
           total  |  20.75 s ( 20.75 s)  20.75 s ( 20.75 s)         1



![png](README_files/README_28_1.png)


    Deceased Total: 3591

