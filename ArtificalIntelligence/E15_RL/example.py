import random

# initialization
Q = []
for i in range(6):
    Q.append([0] * 6)

R = [[-1,-1,-1,-1,0,-1],
     [-1,-1,-1,0,-1,100],
     [-1,-1,-1,0,-1,-1],
     [-1,0,0,-1,0,-1],
     [0,-1,-1,0,-1,100],
     [-1,0,-1,-1,0,100]]

valid_actions = {0:[4],
    1:[3,5],
    2:[3],
    3:[1,2,4],
    4:[0,3,5],
    5:[1,4,5]}

gamma = 0.8

# Q-Learning
for epoch in range(10000):
    state = random.randint(0,5)
    while True: # at least do one action
        # randomly select a action and get to next state
        next_state = random.sample(valid_actions[state],1)[0]
        # get the maxQ of next state
        maxQ = max([Q[next_state][action] for action in valid_actions[next_state]])
        # update current state's Q
        Q[state][next_state] = R[state][next_state] + gamma * maxQ
        # get to next state
        state = next_state
        if state == 5:
            break

# output Q matrix
for row in Q:
    print(*row,sep="\t")

# output path
state = 2
path = [state]
while not state == 5:
    qval, action = max([(Q[state][action],action) for action in valid_actions[state]])
    path.append(action)
    state = action
print(*path,sep=" -> ")