# Chen Hongzheng chenhzh37@mail2.sysu.edu.cn

# Part C: Finding the right amount to save away
# Initialization
semi_annual_raise = 0.07
return_rate = 0.04
portion_down_payment = 0.25
total_cost = 1000000
downpay_cost = total_cost*portion_down_payment
annual_salary = int(input("Enter the starting salary: "))

def cal(rate):
	salary = annual_salary
	current_savings = 0
	for i in range(1,37):
		current_savings += current_savings*return_rate/12 + salary/12*rate
		if (i % 6 == 0):
			salary *= (1+semi_annual_raise)
	return current_savings

# Binary search
step = 0
l, r = 0, 10000
while (l + 1 < r):
	step += 1
	mid = (l+r)//2
	savings = cal(mid/10000)
	# print("{},{},{}".format(l,r,mid))
	if (savings < downpay_cost-100):
		l = mid
	elif (savings > downpay_cost+100):
		r = mid
	else:
		break

# Output
if (l + 1 < r):
	print("Best savings rate: {}".format(mid/10000))
	print("Steps in bisection search: {}".format(step))
else:
	print("It is not possible to pay the down payment in three years.")