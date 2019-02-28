# Problem Set 4A: Permutations of a string
# Name: Chen Hongzheng 17341015
# E-mail: chenhzh37@mail2.sysu.edu.cn

def get_permutations(sequence):
    '''
    Enumerate all permutations of a given string

    sequence (string): an arbitrary string to permute. Assume that it is a
    non-empty string.  

    You MUST use recursion for this part. Non-recursive solutions will not be
    accepted.

    Returns: a list of all permutations of sequence

    Example:
    >>> get_permutations('abc')
    ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

    Note: depending on your implementation, you may return the permutations in
    a different order than what is listed here.
    '''
    if len(sequence) == 1:
        return [sequence]
    head = sequence[0]
    tail = sequence[1:]
    res = []
    for word in get_permutations(tail):
        for i in range(0,len(word)+1):
            if i == 0:
                res.append(head + word)
            elif i == len(word)+1:
                res.append(word + head)
            else:
                res.append(word[:i] + head + word[i:])
    return sorted(res)

if __name__ == '__main__':
#    #EXAMPLE
    example_input = 'abc'
    print('Input:', example_input)
    print('Expected Output:', ['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])
    print('Actual Output:', get_permutations(example_input),"\n")
    
#    # Put three example test cases here (for your sanity, limit your inputs
#    to be three characters or fewer as you will have n! permutations for a 
#    sequence of length n)

    example_input = 'p'
    print('Input:', example_input)
    print('Expected Output:', ['p'])
    print('Actual Output:', get_permutations(example_input),"\n")

    example_input = 'pqrs'
    print('Input:', example_input)
    print('Expected Output:', ['pqrs', 'pqsr', 'prqs', 'prsq', 'psqr', 'psrq', 'qprs', 'qpsr', 'qrps', 'qrsp', 'qspr', 'qsrp', 'rpqs', 'rpsq', 'rqps', 'rqsp', 'rspq', 'rsqp', 'spqr', 'sprq', 'sqpr', 'sqrp', 'srpq', 'srqp'])
    print('Actual Output:', get_permutations(example_input),"\n")

    example_input = 'kl'
    print('Input:', example_input)
    print('Expected Output:', ['kl', 'lk'])
    print('Actual Output:', get_permutations(example_input),"\n")