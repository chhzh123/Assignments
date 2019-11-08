import sys, time, os
import pkuseg

# stage 1
seg = pkuseg.pkuseg()
words = {}

for i in range(1,1001):
	file = open("news/{}.txt".format(str(i)),"r",encoding="utf-8")
	intext = file.read()
	while '\n\n' in intext:
		intext = intext.replace('\n\n','\n')
	text = seg.cut(intext)
	for word in text:
		cnt = words.get(word,-1)
		if cnt == -1:
			words[word] = 1
		else:
			words[word] += 1
	res = " ".join(text)
	res = res.replace('。 ','。\n').replace('。\n” ','。 ”\n')
	outfile = open("dict/{}.txt".format(str(i)),"w",encoding="utf-8")
	outfile.write(res)
	if i % 100 == 0:
		print("Finish {}/1000".format(str(i)),flush=True)

outfile = open("dict/dict_tmp.txt","w",encoding="utf-8")
outfile.write(str(words))

# stage 2
infile = open("dict/dict_tmp.txt",encoding="utf-8")
d = eval(infile.read())
newd = sorted(d,key=d.__getitem__,reverse=True)
outfile = open("dict/dict.txt","w",encoding="utf-8")
for punc in ['，','。','、','“','”','：','（','）','(',')','；','"','《','》','？','+','/',':','・','!','！','-','’','‘','=','<','>','_','……','[',']',';','|','×']:
	try:
		newd.remove(punc)
	except:
		pass
for i,word in enumerate(newd,1):
	outfile.write("{} {} {}\n".format(i,word,d[word]))