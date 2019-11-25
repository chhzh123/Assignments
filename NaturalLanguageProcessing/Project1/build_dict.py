import sys, time, os
import jieba

"""
# Stage 1
Cut Chinese words
"""

news_paths = ["news","news_supplement"]
output_path = "dict_jieba"

cnt = 0
for news_path in news_paths:
	for file_name in os.listdir(news_path):
		if file_name == "dict.txt":
			continue
		cnt += 1
		file = open("{}/{}".format(news_path,file_name),"r",encoding="utf-8")
		intext = file.read()
		while '\n\n' in intext: # remove empty lines
			intext = intext.replace('\n\n','\n')
		text = jieba.lcut(intext,cut_all=False)
		res = " ".join(text)
		res = res.replace('。 ','。\n').replace('。\n” ','。 ”\n') # seperate sentences
		outfile = open("{}/{}.txt".format(output_path,cnt),"w",encoding="utf-8")
		outfile.write(res)
		if cnt % 100 == 0:
			print("Finish {}".format(cnt),flush=True)

sys.exit()