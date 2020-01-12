import sys
import matplotlib.pyplot as plt

file_num = int(sys.argv[1])
draw_every = int(sys.argv[2])

cnt = 0
loss = 0
losses = []
with open("log/seq2seq-{:>02d}.log".format(file_num),"r") as file:
	for i,line in enumerate(file):
		if "Epoch" not in line:
			cnt += 1
			continue
		loss += float(line.split("Loss: ")[-1])
		if (i-cnt) % draw_every == 0:
			losses.append(loss/draw_every)
			loss = 0

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()