"""
Question #2
    You should finish the code here.
"""

class Solution(object):
    """
    TODO: Follow the instructions in the lecture notes.
    """
    def reverse_numbers(self, x):
        if x > 0:
            return int(str(x)[::-1])
        else:
            return (-1) * int(str((-1) * x)[::-1])

    def third_maximum_number(self, l):
        sorted_l = sorted(list(set(l)),reverse=True)
        return sorted_l[0] if len(sorted_l) < 3 else sorted_l[2]