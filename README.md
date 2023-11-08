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
from helpers import *
import casadi

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

N = 5

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

opti.minimize(casadi.sumsqr(i))

for k in range(N):
  opti.subject_to(x[:,k + 1] == f(x[:,k], u[:,k]))
opti.subject_to(opti.bounded(0,u,u_max))
opti.subject_to(opti.bounded(0,casadi.sum1(u),u_max))
opti.subject_to(x[:,0] == x_0)

opti.set_value(x_0, X_0)

opti.solver('sqpmethod', {
  'qpsol':'qrqp',
#   # 'print_header':False,
#   # 'print_iteration':False,
#   # 'print_time':False,
#   # 'qpsol_options':{
#   #   'print_header':False,
#   #   'print_iter':False,
#   #   'print_info':False,
#   # }
})

sol = opti.solve()

t_grid = range(N + 1)
x_grid = sol.value(x)
u_grid = np.hstack((sol.value(u), np.nan*np.ones((n_a,1))))
plot(t_grid, x_grid, u_grid, discrete=True)
```

    Opti {
      instance #19
      #variables: 5 (nx = 174)
      #parameters: 4 (np = 24)
      #constraints: 8 (ng = 179)
      CasADi solver needs updating.
    }
    -------------------------------------------
    This is casadi::QRQP
    Number of variables:                             174
    Number of constraints:                           179
    Number of nonzeros in H:                         396
    Number of nonzeros in A:                         804
    Number of nonzeros in KKT:                      2321
    Number of nonzeros in QR(V):                   10740
    Number of nonzeros in QR(R):                   15412
    -------------------------------------------
    This is casadi::Sqpmethod.
    Using exact Hessian
    Number of variables:                             174
    Number of constraints:                           179
    Number of nonzeros in constraint Jacobian:       804
    Number of nonzeros in Lagrangian Hessian:        396
    
    iter      objective    inf_pr    inf_du     ||d||  lg(rg) ls    info
       0   0.000000e+00  1.06e+06  0.00e+00  0.00e+00       -  0  - 
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
        0    30         0   1.1e+06   329  2.2e-308    30         0   169         0  
        1    29         0   1.1e+06   329  2.2e-308    30         0   168         1  Enforced lbz for regularity, i=319
        2    28         0   1.1e+06   329  2.2e-308    30         0    26         1  Enforced lbz for regularity, i=318
        3    27         0   1.1e+06   329  2.2e-308    30         0   162         1  Enforced lbz for regularity, i=314
        4    26         0   1.1e+06   329  2.2e-308    30         0   170         1  Enforced lbz for regularity, i=312
        5    25         0   1.1e+06   329  2.2e-308    30         0   170         1  Enforced lbz for regularity, i=313
        6    24         0   1.1e+06   329  2.2e-308    30         0   160         1  Enforced lbz for regularity, i=320
        7    23         0   1.1e+06   329  2.2e-308    30         0   160         1  Enforced lbz for regularity, i=304
        8    22         0   1.1e+06   329  2.2e-308    30         0   160         1  Enforced lbz for regularity, i=305
        9    21         0   1.1e+06   329  2.2e-308    30         0   160         1  Enforced lbz for regularity, i=309
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
       10    20         0   1.1e+06   329  2.2e-308    30         0   161         1  Enforced lbz for regularity, i=310
       11    19         0   1.1e+06   329  2.2e-308    30         0   165         1  Enforced lbz for regularity, i=311
       12    18         0   1.1e+06   329  2.2e-308    30         0   165         1  Enforced lbz for regularity, i=299
       13    17         0   1.1e+06   329  2.2e-308    30         0   171         1  Enforced lbz for regularity, i=315
       14    16         0   1.1e+06   329  2.2e-308    30         0   156         1  Enforced lbz for regularity, i=321
       15    15         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=306
       16    14         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=294
       17    13         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=298
       18    12         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=303
       19    11         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=300
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
       20    11         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=308
       21    10         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=307
       22     8         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=296
       23     7         0   1.1e+06   329  2.2e-308    30         0   167         1  Enforced lbz for regularity, i=302
       24     6         0   1.1e+06   329  2.2e-308    30         0   166         1  Enforced lbz for regularity, i=317
       25     5         0   1.1e+06   329  2.2e-308    30         0   173         1  Enforced lbz for regularity, i=316
       26     4         0   1.1e+06   329  2.2e-308    30         0   173         1  Enforced lbz for regularity, i=295
       27     3         0   1.1e+06   329  2.2e-308    30         0   173         1  Enforced lbz for regularity, i=301
       28     2         0   1.1e+06   329  2.2e-308    30         0   172         1  Enforced lbz for regularity, i=323
       29     1         0   1.1e+06   329  2.2e-308    30   7.5e-18     9         1  Enforced lbz for regularity, i=322
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
       30     0         0   1.1e+06   329   8.7e-12    81      0.39   183         1  Enforced ubz for regularity, i=297
       31     0   3.3e+06   8.1e-10   291   5.2e-10   150      0.39   183         1  Converged
       1   3.257846e+06  1.01e+03  4.07e+02  1.06e+06       -  1  - 
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
        0     0         0     1e+03   181   4.1e+02    37      0.31   205         0  
        1     0   3.6e+07   3.1e-11   293       4.6   145   1.2e-07   145         1  Dropped lbz to reduce |du|, i=295
        2     0   3.6e+07       2.3   324       4.6   145      0.31   205   1.9e-07  Enforcing ubz, i=324
        3     0   3.6e+07   1.1e-12   276       4.2   147   7.5e-08     9         1  Dropped ubz to reduce |du|, i=297
        4     0   3.6e+07       2.1   295       4.2   147      0.31   205    0.0046  Enforcing ubz, i=295
        5     0   3.6e+07   7.3e-12   261       3.6   144   6.5e-08     9         1  Dropped lbz to reduce |du|, i=294
        6     0   3.6e+07       1.8   297       3.6   144      0.31   205   1.5e-07  Enforcing lbz, i=297
        7     0   3.6e+07   7.3e-12   213      0.74   151   4.4e-09   151         1  Dropped lbz to reduce |du|, i=301
        8     0   3.6e+07      0.37   301      0.74   151      0.31   205   0.00098  Enforcing ubz, i=301
        9     0   3.6e+07   1.8e-11   223      0.68   150   4.1e-09   150         1  Dropped lbz to reduce |du|, i=300
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
       10     0   3.6e+07      0.34   325      0.68   150      0.31   205   6.1e-09  Enforcing ubz, i=325
       11     0   3.6e+07   1.1e-11   273      0.12   157     2e-10   157         1  Dropped lbz to reduce |du|, i=307
       12     0   3.6e+07     0.059   307      0.12   157      0.31   205   0.00025  Enforcing ubz, i=307
       13     0   3.6e+07   3.5e-11   247      0.11   156   1.8e-10   156         1  Dropped lbz to reduce |du|, i=306
       14     0   3.6e+07     0.055   326      0.11   156      0.31   205   2.4e-10  Enforcing ubz, i=326
       15     0   3.6e+07   1.4e-11   259     0.016   163   2.3e-06   163         1  Dropped lbz to reduce |du|, i=313
       16     0   3.6e+07    0.0079   313     0.016   163      0.31   205   8.3e-05  Enforcing ubz, i=313
       17     0   3.6e+07   1.8e-11   283     0.014   162   2.3e-06   162         1  Dropped lbz to reduce |du|, i=312
       18     0   3.6e+07    0.0072   327     0.014   162      0.31   205   7.8e-12  Enforcing ubz, i=327
       19     0   3.6e+07   1.4e-11   259   3.6e-12    37      0.31   205         1  Converged
       2   3.935762e+07  4.20e+02  7.72e+02  2.27e+05       -  1  - 
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
        0     0         0   4.2e+02   271   7.7e+02    49      0.32   205         0  
        1     0  -6.2e+06   5.7e-12   265       1.8   157   1.1e-05   156         1  Dropped ubz to reduce |du|, i=307
        2     0  -6.2e+06   5.1e-11   234      0.52   163   1.6e-06   156         1  Dropped ubz to reduce |du|, i=313
        3     0  -6.3e+06      0.26   312      0.42   163   1.1e-05   156       0.2  Enforcing ubz, i=312
        4     0  -6.3e+06   1.5e-11   247   7.3e-12    37   1.1e-05   156         1  Converged
       3   3.311891e+07  1.73e+01  2.78e+01  7.47e+04       -  1  - 
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
        0     0         0        17   271        28    61   8.8e-06   156         0  
        1     0  -1.9e+04   3.3e-12   265   4.1e-12    54   8.8e-06   156         1  Converged
       4   3.310032e+07  6.29e-03  1.25e-01  2.97e+03       -  1  - 
     Iter  Sing        fk      |pr|   con      |du|   var     min_R   con  last_tau  Note
        0     0         0    0.0063   277      0.13    55   8.8e-06   156         0  
        1     0        30   1.2e-14   216   3.6e-12    36   8.8e-06   156         1  Converged
       5   3.310035e+07  6.99e-09  1.50e-07  3.90e+00       -  1  - 
    MESSAGE(sqpmethod): Convergence achieved after 5 iterations
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
              QP  | 200.00ms ( 40.00ms) 199.34ms ( 39.87ms)         5
      linesearch  |        0 (       0)  92.00us ( 18.40us)         5
          nlp_fg  |        0 (       0)  71.00us ( 14.20us)         5
      nlp_hess_l  |        0 (       0) 388.00us ( 77.60us)         5
      nlp_jac_fg  |   1.00ms (166.67us) 731.00us (121.83us)         6
           total  | 201.00ms (201.00ms) 201.37ms (201.37ms)         1



![png](README_files/README_26_1.png)

