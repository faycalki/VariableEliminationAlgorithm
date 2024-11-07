# Introduction
This algorithm produces a probability distribution over a query variable. There are several limitations to the algorithm, as is denoted in the current code base under "TODO".

## Features 
- Performs variable elimination given a list of factors (CPTs) and a specified elimination order. 
- Supports restricting factors based on evidence. 
- Handles multiplication, marginalization, normalization of factors. 
- Designed for discrete variables in a Bayesian Network.

# Operating the algorithm (usage) and Implementation

There are implementations for all the key operations:
* Restriction (narrowing down the factors to particular random variable values)
* Multiplication (Pointwise product -- also known as Hadamard Product)
* Sum out (Also known as marginalization of a random variable)
* Normalization

The preconditions are set for the argument factors, evidence, and elimination order as follows:

|               | Factor                                                                                                                                                                                                                                                                                                                   | Evidence                                                                                                           | Elimination Order                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Precondition  | A list consisting of elements $e \in$ (`numpy.ndarray` objects where each random variable $X_1, \dots, X_n$ is in its own dimension). Furthermore, each $e$ must have the same number of dimensions and corresponding random variables to those dimensions (that is, the random variables are uniform to the dimensions) | A list consisting of elements $e \in$ (`tuple`. Where each `tuple` consists of (`dimension`, `restriction_value`)) | A list of elements dimension (random variables) corresponding to the random variables of the `factors` argument to eliminate in left-to-right order. |
| Postcondition | Probability Distribution over Query variable                                                                                                                                                                                                                                                                             | All evidence is exhausted to improve performance.                                                                  | Elimination order is followed during execution of algorithm.                                                                                         |

The algorithm produces a probability distribution over a query variable by following the Variable Elimination Algorithm. In short, it performs the following steps:
1. Restricts the random variables according to the evidence variables presented in the form described.
2. Performs Hadamard Product multiplication on all the restricted factors
3. Performs Marginalization to essentially drop a Random Variable from the probability distribution
4. Repeats 2-3 until the factor is reduced to only the query variable.
5. Normalizes the factor, therefore constructing a probability distribution over the query variable and returns it.


In order to use this, you'll have to understand the following function
- **`variable_elimination(factor_list, elimination_order, evidence)`**:  
  This is the main function that runs the variable elimination algorithm.
  - `factor_list`: List of factors (numpy arrays) representing the conditional probability distributions (CPTs).
  - `elimination_order`: List specifying the order in which variables should be eliminated.
  - `evidence`: List of tuples, each containing a variable and its observed value (e.g., `[(X, x), (Y, y)]`).
  - **Remark**: the query variable is implicitly encoded in the elimination order.
### Usage Example
```python
import numpy as np
from variable_elimination import variable_elimination

# Define the factors (CPTs) as numpy arrays. Ensure they have the same dimensionality.
factor_b = np.zeros((2, 1, 1, 1, 1))
factor_b[0, 0, 0, 0, 0] = 0.999
factor_b[1, 0, 0, 0, 0] = 0.001

factor_e = np.zeros((1, 2, 1, 1, 1))
factor_e[0, 0, 0, 0, 0] = 0.998
factor_e[0, 1, 0, 0, 0] = 0.002

# Add more factors as needed

factor_set = [factor_b, factor_e, ...]  # List of factors
evidence = [(4, 1), (3, 1)]  # Example evidence: M = 1, J = 1

# Specify the elimination order as you'd like.
elimination_order = [0, 2, 3, 4]

# Run the variable elimination algorithm
query_variable_distribution = variable_elimination(factor_set, elimination_order, evidence)

```

# Discussion of test cases
A test suite case is provided with the following inputs and results

```python
# test_network_1
# Variables correspond to axes: B->0, E->1, A->2, J->3, M->4  
# Indices correspond to values: False->0, True->1  
factor_b = np.zeros((2, 1, 1, 1, 1))  
factor_b[0, 0, 0, 0, 0] = 0.999  
factor_b[1, 0, 0, 0, 0] = 0.001  
factor_e = np.zeros((1, 2, 1, 1, 1))  
factor_e[0, 0, 0, 0, 0] = 0.999  
factor_e[0, 1, 0, 0, 0] = 0.001  
factor_bea = np.zeros((2, 2, 2, 1, 1))  
factor_bea[0, 0, 0, 0, 0] = 0.99  
factor_bea[0, 0, 1, 0, 0] = 0.01  
factor_bea[0, 1, 0, 0, 0] = 0.30  
factor_bea[0, 1, 1, 0, 0] = 0.70  
factor_bea[1, 0, 0, 0, 0] = 0.99  
factor_bea[1, 0, 1, 0, 0] = 0.01  
factor_bea[1, 1, 0, 0, 0] = 0.30  
factor_bea[1, 1, 1, 0, 0] = 0.70  
factor_aj = np.zeros((1, 1, 2, 2, 1))  
factor_aj[0, 0, 0, 0, 0] = 0.95  
factor_aj[0, 0, 0, 1, 0] = 0.05  
factor_aj[0, 0, 1, 0, 0] = 0.10  
factor_aj[0, 0, 1, 1, 0] = 0.90  
factor_am = np.zeros((1, 1, 1, 2, 2))  
factor_am[0, 0, 0, 0, 0] = 0.99  
factor_am[0, 0, 0, 0, 1] = 0.01  
factor_am[0, 0, 0, 1, 0] = 0.30  
factor_am[0, 0, 0, 1, 1] = 0.70
```
This results in the following outputs

```
P(E | M = true):  
[[[[[0.99103697]]]  
  
  
  [[[0.00896303]]]]]  
  
P(B | M = true):  
[[[[[0.999]]]]  
  
  
  
 [[[[0.001]]]]]  
  
P(E | J = true, M = true):  # This output is strange
[[[[0.9890838]]]


 [[[0.0109162]]]]
```
Of which, all are correct, except for the one that has a strange output. That one, in particular, may in fact be correct but one will need to verify it by hand.

The second test suite case is
```python
# Variables correspond to axes: B->0, E->1, A->2, J->3, M->4  
# Indices correspond to values: False->0, True->1  
factor_b = np.zeros((2, 1, 1, 1, 1))  
factor_b[0, 0, 0, 0, 0] = 0.999  
factor_b[1, 0, 0, 0, 0] = 0.001  
factor_e = np.zeros((1, 2, 1, 1, 1))  
factor_e[0, 0, 0, 0, 0] = 0.998  
factor_e[0, 1, 0, 0, 0] = 0.002  
factor_bea = np.zeros((2, 2, 2, 1, 1))  
factor_bea[0, 0, 0, 0, 0] = 0.99  
factor_bea[0, 0, 1, 0, 0] = 0.01  
factor_bea[0, 1, 0, 0, 0] = 0.30  
factor_bea[0, 1, 1, 0, 0] = 0.70  
factor_bea[1, 0, 0, 0, 0] = 0.99  
factor_bea[1, 0, 1, 0, 0] = 0.01  
factor_bea[1, 1, 0, 0, 0] = 0.30  
factor_bea[1, 1, 1, 0, 0] = 0.70  
factor_aj = np.zeros((1, 1, 2, 2, 1))  
factor_aj[0, 0, 0, 0, 0] = 0.95  
factor_aj[0, 0, 0, 1, 0] = 0.05  
factor_aj[0, 0, 1, 0, 0] = 0.10  
factor_aj[0, 0, 1, 1, 0] = 0.90  
factor_am = np.zeros((1, 1, 2, 1, 2))  
factor_am[0, 0, 0, 0, 0] = 0.99  
factor_am[0, 0, 0, 0, 1] = 0.01  
factor_am[0, 0, 1, 0, 0] = 0.30  
factor_am[0, 0, 1, 0, 1] = 0.70
```

of which matches the expected outputs in every case
```
P(E | M = true):  
[[[[[0.94476871]]]  
  
  
  [[[0.05523129]]]]]  
  
P(B | M = true):  
[[[[[0.999]]]]  
  
  
  
 [[[[0.001]]]]]  
  
P(E | J = true, M = true):  
[[[[[0.88487299]]]  
  
  
  [[[0.11512701]]]]]
```

The outputs are mostly as expected. Therefore, the algorithm performs its job accordingly.
# Further along (improvements for performance)
Different orderings will yield eventually the same valid probability distribution over the query variable(s). However, different orderings can result in wildly different computational time requirements based on the factors structure. This is due to the fact that different elimination orderings can result in the creation of additional *intermediate factors* during the calculation. 

Generally, the time and space complexity of the variable elimination algorithm is determined by the size of the *largest factor* that is constructed during the execution of the algorithm -- which is determined by the order of eliminating the random  variables and by the structure of the underlying Bayesian Network.

if we improved our algorithm in order to choose the minimal largest factor to construct every time, then we'd increase our performance.

A good heuristic that can be implemented is actually a greedy one, and it is as follows: 
* Choose to eliminate the random variable that would minimize the size of the *next* factor to be constructed. 

The ordering generated by such a heuristic would generally be considered good, because finding the optimal ordering is usually a *intractable* problem.

Another important optimization that can be performed is removing redundancies: this is allowed because we can remove *any* leaf node in the underlying Bayes Net that isn't the query variable nor an evidence variable.

Therefore, if a node is determined to be a non-query and a non-evidence node, and also happens to be a leaf node, then it should be removed. Performing this operation may result in additional nodes becoming leaf nodes. The leaf nodes (and the leaf nodes generated after the removal of those preceding leaf nodes) should all be removed in order to reduce the time complexity and space complexity of the algorithm and underlying Bayes net, as they are irrelevant to the query.