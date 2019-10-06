task = eval(open("tasks.json","r").read())["17341015"]
task.sort()
gaokao_url = eval(open("gaokao_url.txt","r").read())
official_url = eval(open("official_url_new.txt","r").read())
school_chinese = eval(open("school_chinese.txt","rb").read())
print(len(task),len(gaokao_url),len(official_url),len(school_chinese))

for i in range(len(gaokao_url)):
	if gaokao_url[i] == None:
		print(task[i],school_chinese[i])

print()
cnt = 0
for i in range(len(official_url)):
	if official_url[i] == None:
		cnt += 1
	print(task[i],school_chinese[i],official_url[i])
print("Total: {}".format(cnt))

import os, sys
folder = sys.argv[1]
for i in range(len(task)):
	if official_url[i] == None:
		continue
	path = folder + "/" + str(task[i]) + "-" + school_chinese[i]
	file = open(path + "/README.md","w")
	context = "## " + str(task[i]) + "-" + school_chinese[i] + "\n\n"
	context += "* 学校官网： <" + official_url[i] + ">\n"
	if os.path.isfile(path + "/index.html"):
		context += "* 使用说明：\n\t1. `index.html` 为学校官网首页\n"
		if official_url[i][7:][-1] == "/":
			url = official_url[i][7:-1]
		else:
			url = official_url[i][7:]
		context += "\t2. 学校官网也可以从 `{}/index.html` 进行访问\n".format(url)
		context += "\t3. 若网页中的超链接(`href`)采用相对链接的方式，则可以直接离线访问文件夹中的其他子网页；否则只能进入每一个文件夹逐一进行查看"
	else:
		context += "* Error: 本学校网页无法正常打开！"
	file.write(context)