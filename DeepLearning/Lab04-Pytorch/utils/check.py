import torch
import time

def check_cross_entropy_loss(criterion):
    print ("Running check cross entropy loss\n")
    rand_input = torch.rand((1000,2))
    rand_target = torch.randint(2, (1000, ))
    time_a = time.time()
    loss = criterion(rand_input, rand_target)
    time_b = time.time()
    loss_official = torch.nn.functional.cross_entropy(rand_input, rand_target)
    time_c = time.time()
    period1 = time_c-time_b
    period2 = time_b-time_a
    error = (loss-loss_official).abs().item()
    print ("Official implementation time usage: {}".format(period1))
    print ("Your implementation time usage: {}".format(period2))
    print ("Error between two implementations: {}".format(error))
    print ()
    if error > 1e-5:
        print("The error between two implementations is too large, check your code!")
        exit(1)

def check_mse_loss(criterion):
    print ("Running check mean squared error loss\n")
    rand_input = torch.rand((1000,2))
    rand_target = torch.randint(2, (1000, ))
    time_a = time.time()
    loss = criterion(rand_input, rand_target)
    time_b = time.time()
    rand_target_tensor = torch.nn.functional.one_hot(rand_target, num_classes=2).float()
    loss_official = torch.nn.functional.mse_loss(rand_input, rand_target_tensor)
    time_c = time.time()
    period1 = time_c-time_b
    period2 = time_b-time_a
    error = (loss-loss_official).abs().item()
    print ("Official implementation time usage: {}".format(period1))
    print ("Your implementation time usage: {}".format(period2))
    print ("Error between two implementations: {}".format(error))
    print ()
    if error > 1e-5:
        print("The error between two implementations is too large, check your code!")
        exit(1)