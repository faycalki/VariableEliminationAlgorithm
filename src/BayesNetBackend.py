# Author: Faycal Kilali
# Variable Elimination Algorithm
# Implementation requires taking a set of factors in and an elimination ordering of variables as arguments.
# Output: Probability distribution over query variable (that is, the probability distribution of the query variable). Normalized.
# TODO: Test if this works with multiple query variables, and update documentation.
# TODO: Expand to include the ability to have multiple restriction values for any particular random variable.
# TODO: Further expand the multiplication function to handle cases where the dimensions of the factors are non-uniform (not in the same order) and if the number of dimensions also differs should be handled by the Variable_Elimination_Algroithm function.
# TODO: Add the ability to graph the Bayesian Network as well as the ability to generate the appropriate factors from the Bayesian Network for inference purposes.
# TODO: Deduce optimal or near-optimal elimination order. This can be performed by using known good heuristics.
import numpy as np



def variable_elimination(factor_list, elimination_order, evidence):
    '''
    This function preasumes that the factors are already produced. This algorithm does not produce the factors by itself, it simply takes them as a set and an order to eliminate variables within the factors.
    Where the elimination ordering is the order in which the variables are eliminated, and is fed as an argument to the function as a list of dimensions, in the order from left to right of the list (elimination order) of which dimensions to eliminate. This is actually the index of the dimension.

    :usage: Provide a uniform

    :definition: In linear Algebra, a matrix M of form (n, m) has size nm. In here, the dimensions count does *not* include the rows as a particular dimension. E.g; a column vector is a single dimension in numpy. A two-column matrix is also a two-dimensional matrix.
    :note: the left-over variable that isn't included in the elimination order will be the query variable, implicitry encoded.
    :param factor_list: the list of factors. Each element f in factor_set will have the same number of dimensions.
    :param elimination_order: the order in which to eliminate variables in the factors
    :param evidence: tuples of random variables with their restriction to set, e.g; [(X,x), (Y,y), (Z,z)]. Limited to a single restriction per random variable.
    :return: the probability distribution over the query variable, relative to the restrictions
    :note: Dimensions of size 1 essentially mean the variable has no effect on the factor values, but it’s still technically present in the table.
    :note: By setting unused dimensions to 1, all indices for that dimension will be 0 (since there’s only one index, 0). However, the factor’s values remain unchanged by this "dummy" dimension.
    :precondition: the factors in factor_set have the same number of dimensions and the same order of dimensions, and their dimensionality is uniform. For example, For example, if a factor does not depend on variable X, we can imagine that it has a trivial dimension for X with size 1, so it won’t affect the outcome when performing operations.
    :precondition: the evidence variables have already been eliminated.
    TODO: Have a list of tuples of random dimensions with their value restriction to restrict to.
    '''

    # For each variable in elimination ordering...
    # 0. Restrict if evidence variables are provided (that is, the variables we'll eliminate) -- this is done by the other function
    # 1. Multiply out the factors that have the hidden variable V only when you need to sum them out
    # 2. Sum them out
    # 3. Repeat (1-2) until you are down to the query variable Q, with the relevant restrictions
    # 4. Normalize the factor that consists of only query variable of non-size 1.
    # 5. Return the factor (probability distribution over query variable)

    # Creating a deep copy of the factors in order to prevent potential issues with modifying the original factors. This is primarily due to keep order of factors, although its possible to keep order in other ways.
    factors = [f.copy() for f in factor_list]

    # Apply restriction to the factors if there is evidence for the variable
    if evidence is not None:
        new_factors = []
        for factor in factors:
            temp_factor = factor.copy()
            for tuple in evidence:
                if factor.shape[tuple[0]] > 1:  # Only restrict if dimension is of size > 1, for optimization purposes.
                    temp_factor = restrict(temp_factor, tuple) # Sequentially restrict
            new_factors.append(temp_factor)
        factors = new_factors

    # Apply the rest of the Variable  Elimination Algorithm
    for random_variable_to_eliminate in elimination_order:
        # Tagging factors that include the random variable (dimension)
        tag = [factor for factor in factors if factor.shape[random_variable_to_eliminate] > 1]

        # Multiply all tagged factors
        if tag:
            product_factor = tag[0]
            for factor in tag[1:]: # For factors starting after the factors from index 1 onwards by slicing.
                product_factor = multiplication(product_factor, factor)

            # Sum out the variable
            marginalized_factor = marginalize(product_factor, random_variable_to_eliminate)

            # Update factor list
            factors = [f for f in factors if not any(np.array_equal(f, t) for t in tag)]
            factors.append(marginalized_factor)

    # Making sure that all the remaining factors are multiplied together
    if len(factors) > 1:
        result = factors[0]
        for f in factors[1:]:
            result = multiplication(result, f)
    else:
        result = factors[0]

    # Normalize the remaining factor (query variable)
    normalized_query = normalize(result)
    return normalized_query


def generate_factor(shape):
    """
    Generates a factor with the specified shape
    For the purpose of variable elimination's multiplication aspect, the factor generated is a matrix with the same dimensions
    :param shape: the shape of the matrix (dimensionality)
    :return: the generated factor, filled with zeros
    """
    f_new = np.zeros(shape)
    return f_new


def restrict(factor, restriction):
    """
    Restricts the factor to only include entries along the specified dimension that match a specified value.

    :param factor: The factor to restrict (numpy array)
    :param restriction: Tuple of (dimension_to_restrict, value_to_restrict)
    :return: A restricted factor with dimensions consistent with the original
    """
    dimension_to_restrict, value_to_restrict = restriction

    # Check if the dimension we want to restrict has a size of 1, if it is, no need to do anything
    if factor.shape[dimension_to_restrict] == 1:
        return factor

    # Create slice objects for all dimensions
    slices = [slice(None)] * factor.ndim
    slices[dimension_to_restrict] = value_to_restrict

    # Applying the restriction to our factor
    indexing_slicing_objects_in_tuple_form = tuple(slices) # slices is a list of slice objects for each dimension. Tuple converts the list to components elements in the list. This will be used to index the array in order to construt a restricted array.
    restricted_factor = factor[indexing_slicing_objects_in_tuple_form]

    # Restore the restricted dimension as size 1
    restricted_factor = np.expand_dims(restricted_factor, axis=dimension_to_restrict)

    return restricted_factor


def multiplication(factor1, factor2):
    """
    Multiplies the two parameter factors using the Hadamard Product definition (Pointwise multiplication)
    :param factor1: first factor
    :param factor2: second factor
    :return: product of factor1 and factor 2 using the Hadamard Product definition
    :precondition: factor1 and factor2 have the same shape
    :limitation: this implementation requires both factors to share the same variable(s) and no other variables.

    """
    #TODO: not sure why this exception check breaks
    #if (factor1.shape != factor2.shape):
       # raise Exception("Factors must have the same shape! (Shape of factor1: " + str(factor1.shape) + ", Shape of factor2: " + str(factor2.shape) + ")")
    return np.multiply(factor1, factor2) # Hadamard Product

def normalize(factor):
    """
    Normalizes the matrix provided, then returns it
    :param factor: the array to be normalized
    :return: the normalized factor
    """
    normalization_factor = np.sum(factor, keepdims=True)
    return factor / normalization_factor
def marginalize(factor, dimension_to_marginalize):
    """
    Marginalize a specified column from a multi-dimensional factor array.

    This function takes a factor array and marginalizes (sums out) the specified
    dimensions (random variable) across the remaining dimensions. The resulting
    array is returned without normalization. Note that marginalization is synonymous
    with summing out.


    :param factor: A NumPy array representing the factor from which to marginalize a variable.
                   The shape of the array should have at least two dimensions, and the column
                   specified by `column_to_marginalize` must have more than one entry.
    :type factor: numpy.ndarray
    :param dimension_to_marginalize: The index of the (dimension) to marginalize out.
                                   This should be an integer corresponding to the axis of the factor
                                   array that is to be summed out.
    :type dimension_to_marginalize: int

    :raises ValueError: If the size of the dimension specified by `column_to_marginalize`
                        is 1 or less, indicating that it cannot be marginalized.

    :return: A new NumPy array representing the marginalized factor with the specified column
             summed out. The shape of the returned array will be the same as the original factor
             except with the size of the marginalized dimension reduced by one.
    :rtype: numpy.ndarray

    :impl: The normalization factor is not implemented in this function. It is deferred to be used from a different function in order to avoid having to calculate it multiple times.
    :limitation: This function only works for multi-valued discrete values for the column specified by `column_to_marginalize`. or binary values. For continuously-valued, you'll need to use the np.sum approach instead.
    """

    # Exception check
    if factor.shape[dimension_to_marginalize] <= 1:
        raise ValueError("Cannot marginalize a dimension with size 1 or less.")

    # Swap the axis to marginalize to be the last axis (last dimension)
    #np.swapaxes(factor, dimension_to_marginalize, np.ndim(factor) - 1)

    # New shape
    #new_shape = factor.shape[:-1]  # Reduce last dimension by 1
    new_shape = factor.shape  # Last dimension not dropped


    # Creating the matrix with the appropriate, new shape.
    f_mar = generate_factor(new_shape)  # Create a new matrix with the adjusted shape, filled with zeros.

    # Alternative Implementation
    f_mar = np.sum(factor, axis=dimension_to_marginalize, keepdims=True) # Sums out the axis specified, marginalizing over it, and keeps the number of dimensions

#    # We'll use our original approach regardless, for now, unless I comment them out and uncomment the above.
#    rows = factor.shape[0]
#    dimensions = np.ndim(factor)
#
#    #columns_indices = tuple(0 for _ in range(dimensions - 2)) # Create a tuple of zeros, excluding the last dimension
#    columns_indices = tuple(0 for _ in range(dimensions - 1)) # Create a tuple of zeros, excluding the first dimension
#    #TODO: Do not drop the dimension! This'll be important for the multiplication later
#    for i in range(0, rows):
#        for j in range(i, rows):
#            if np.array_equal(factor[i, columns_indices],
#                              factor[j, columns_indices]):  # True if the arrays are equivalent!
#                f_mar[i, columns_indices] += factor[i, columns_indices] + factor[j, columns_indices]
#                # Optional: if we can only really add two values, we can cause a break here by returning here to improve performance.
#                # This may potentially be a problematic approach. A better approach may be to have j approach from the second half, and i from the first half.
#                # It is possible this may only work for boolean-valued dimension to sum out.
    return f_mar  # Return the factor, marginalized (summed over) the column specified by column_to_marginalize.




# Helper function to print results
def print_results(elimination_order, result):
    print(f"Results for elimination order: {elimination_order}")
    for i, res in enumerate(result):
        print(f"Factor {i} (after elimination):")
        print(res)
    print("\n" + "="*40 + "\n")


def test_network():
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

    factor_set = [factor_b, factor_e, factor_bea, factor_aj, factor_am]
    evidence = [(4, 1), (3, 1)]  # M = true, # J = TRUE

    # Try different elimination orders
    for order in [[0, 2, 3, 4]]:
        print_results(order, variable_elimination(factor_set, order, evidence))


def test_network_2():
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

    factor_set = [factor_b, factor_e, factor_bea, factor_aj, factor_am]
    evidence = [(4,1), (3,1)]

    # Try different elimination orders
    for order in [(0, 2, 3, 4)]:
        print_results(order, variable_elimination(factor_set, order, evidence))


# Running the test cases
test_network()
#test_network_2()


