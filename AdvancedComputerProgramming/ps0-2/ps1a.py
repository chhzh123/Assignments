###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name: Hongzheng Chen 17341015

from ps1_partition import get_partitions
import time
import sys # used for read command line arguments

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    res = {} # create an empty dictionary
    with open(filename,"r") as file: # avoid exception
        for line in file:
            (name,num) = line.split(',')
            res[name] = int(num) # remember to take int
    return res

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Firstly sort the dict by value
    # The code below will not ruin the original dict
    cow_sorted = sorted(cows.items(),key=lambda item:item[1],reverse=True)
    res = []
    # the input must ensure the biggest cow <= limit
    while (len(cow_sorted) > 0):
        curr_weight = 0
        one_trip = []
        for item in cow_sorted[:]: # copy out!
            if (curr_weight + item[1] <= limit):
                curr_weight += item[1]
                one_trip.append(item[0]) # append the name
                cow_sorted.remove(item)
        res.append(one_trip) # add the last
    return res

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    lst = cows.items() # will not ruin the original dict
    min_num_trip = len(cows)
    min_trip = []
    for partition in get_partitions(lst):
        flag = False # use for test if all trips valid
        for trip in partition:
            weights = sum([item[1] for item in trip]) # sum all weights in one trip
            if (weights > limit):
                flag = True
                break
        if flag:
            continue
        elif (len(partition) < min_num_trip): # find the min trip
            min_num_trip = len(partition)
            min_trip = [[item[0] for item in trip] for trip in partition] # take out all the names
    return min_trip

# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    if (len(sys.argv) == 1):
        cows = load_cows("ps1_cow_data.txt")
        limit = 10
    else: # my test data
        cows = load_cows("ps1_cow_data-my.txt")
        limit = 15

    start = time.time()
    res = greedy_cow_transport(cows,limit)
    num_trip_greedy = len(res)
    end = time.time()
    print("Greedy method: {}, Time: {}s".format(num_trip_greedy,end-start))
    print(res)

    print()

    start = time.time()
    res = brute_force_cow_transport(cows,limit)
    num_trip_bf = len(res)
    end = time.time()
    print("Brute force: {}, Time: {}s".format(num_trip_bf,end-start))
    print(res)

if __name__ == '__main__':
    compare_cow_transport_algorithms()