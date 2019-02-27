# Chen Hongzheng chenhzh37@mail2.sysu.edu.cn

# Part B: Saving, with a raise
# Input
print(">>>")
annual_salary = int(input("Enter your annual salary: "))
portion_saved = float(input("Enter the percent of your salary to save, as a decimal: "))
total_cost = int(input("Enter the cost of your dream home: "))
semi_annual_raise = float(input("Enter the semi-annual raise, as a decimal: "))
# Calculation
portion_down_payment = 0.25
current_savings = 0
r = 0.04
cnt = 0
while (current_savings < total_cost*portion_down_payment):
	current_savings += current_savings*r/12 + annual_salary/12*portion_saved
	cnt += 1
	if (cnt % 6 == 0):
		annual_salary *= (1+semi_annual_raise)
# Output
print("Number of months: {}".format(cnt))
print(">>>")