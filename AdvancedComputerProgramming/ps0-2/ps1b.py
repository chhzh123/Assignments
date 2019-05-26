###########################
# 6.0002 Problem Set 1b: Space Change
# Name: Hongzheng Chen 1734105

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # Greedy method may not return the optimal solution!!!
    # if (len(egg_weights) == 0):
    #     return 0
    # num_eggs = target_weight // egg_weights[-1]
    # return num_eggs + dp_make_weight(egg_weights[:-1],target_weight-num_eggs*egg_weights[-1])

    # Dynamic Programming
    memo[0] = 0
    for target in range(1,target_weight+1):
        for weight in egg_weights:
            if (target >= weight):
                # DP transition equation
                memo[target] = min(memo.get(target,target),1+memo[target-weight])
    return memo[target_weight]


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n, {}))
    print()

    egg_weights = (1, 5, 11)
    n = 15
    print("Egg weights = (1, 5, 11)")
    print("n = 15")
    print("Expected ouput: 3 (3 * 5 = 15)")
    print("Actual output:", dp_make_weight(egg_weights, n, {}))
    print()