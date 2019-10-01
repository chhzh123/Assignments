import os, sys, time, platform
import pandas as pd
import requests
import urllib.request as urllib2
from urllib.parse import urljoin
from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.keys import Keys
import multiprocessing
import logging

MAX_DEPTH = 3

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

url_file_name = "gaokao_url.txt"
official_url_file_name = "official_url.txt"
sep = "\\" if 'Windows' in platform.system() else "/"

task = eval(open("tasks.json","r").read())["17341015"]
task.sort()
# print(len(task))
# print(task)

df = pd.read_excel("普通高等大学名单.xls",header=2)
# df = pd.read_excel("school.xls",header=2)
# print(df.columns.values)
mask = [True if item in task else False for item in df["序号"]]
school_name = [school for school in df["学校名称"][mask]]
# print(school_name)

if url_file_name not in os.listdir():
	gaokao_url = []
	for (step,school) in enumerate(school_name,1):
		print(step,end=" ")
		url = "https://gaokao.chsi.com.cn/sch/search.do?searchType=1&yxmc=" + school + "&zymc=&sySsdm=&ssdm=&yxls=&yxlx=&xlcc="
		page = requests.get(url,timeout=30)
		soup = BeautifulSoup(page.content,'lxml')
		try:
			td = soup.find("td",attrs={"class":"js-yxk-yxmc"})
			tag = td.contents[1]
			link = tag["href"]
			gaokao_url.append("https://gaokao.chsi.com.cn" + link)
		except:
			gaokao_url.append(None)

	print(gaokao_url)
	outfile = open(url_file_name,"w")
	outfile.write(str(gaokao_url))
else:
	url_file = open(url_file_name,"r")
	gaokao_url = eval(url_file.read())

# print(gaokao_url)

if official_url_file_name not in os.listdir():
# if True:
	official_url = []
	cnt_none = 0
	for (step,url) in enumerate(gaokao_url,1):
		print(step,end=" ")
		if url == None:
			continue
		page = requests.get(url,timeout=30)
		soup = BeautifulSoup(page.content,'lxml')
		try:
			# span = soup.find("span",attrs={"class":"judge-empty"})
			# tag = span.contents[0]
			# link = tag["href"]
			span = soup.find("div",attrs={"class":"msg"})
			tag = span.find("a")
			link = tag["href"]
			official_url.append(link)
		except:
			cnt_none += 1
			official_url.append(None)

	print(official_url)
	outfile = open(official_url_file_name,"w")
	outfile.write(str(official_url))
	print(cnt_none)
else:
	url_file = open(official_url_file_name,"r")
	official_url = eval(url_file.read())

def crawl(pages, school_abbr, school_chinese_name, depth=1, pagefolder="pages"):
	pagefolder += sep + school_chinese_name
	if not os.path.exists(pagefolder):
		os.makedirs(pagefolder)
	try:
		html = urllib2.urlopen(pages[0]).read()
		folder = pagefolder + sep + "index.html"
		with open(folder,"wb") as file:
			file.write(html)
		logger.info("Saved {}".format(folder))
		# if not os.path.isfile(folder):
		# 	with open(folder,"wb") as file:
		# 		file.write(html)
		# 	logger.info("Saved {}".format(folder))
	except:
		logger.info("Cannot create {}".format(folder))
	indexed_url = []
	curr_depth = 0
	while len(pages) != 0:
		curr_depth += 1
		if curr_depth > MAX_DEPTH:
			break
		new_pages = []
		for page in pages:
			if page not in indexed_url:
				indexed_url.append(page)
				try:
					html = urllib2.urlopen(page).read()
				except:
					logger.info("Could not open {}".format(page))
					continue
				filename = page[7:].split("/")
				if filename[-1] == "" or filename[-1][-3:] == ".cn" or filename[-1][-4:] == ".com":
					folder = pagefolder + sep + filename[0]
					try:
						if not os.path.exists(folder):
							os.makedirs(folder)
						folder += "\\index.html"
						if not os.path.isfile(folder):
							with open(folder,"wb") as file:
								file.write(html)
							logger.info("Saved {}".format(folder))
					except:
						logger.info("Cannot create {}".format(folder))
				else:
					for i,path in enumerate(filename):
						folder = pagefolder + sep + sep.join(filename[:i+1])
						if i == len(filename) - 1:
							if not os.path.isfile(folder):
								try:
									with open(folder,"wb") as file:
										file.write(html)
									logger.info("Saved {}".format(folder))
								except:
									logger.info("Cannot save {}".format(folder))
						else:
							try:
								if not os.path.exists(folder):
									os.makedirs(folder)
							except:
								logger.info("Cannot create {}".format(folder))
				# get next urls
				soup = BeautifulSoup(html,'lxml')
				links = soup.find_all("a") # find all sub links
				for link in links:
					if "href" in dict(link.attrs):
						rel_path = link['href']
						tmp_url = urljoin(page,link['href'])
						if tmp_url.find("'") != -1 or tmp_url.find(school_abbr) == -1:
							continue
						tmp_url = tmp_url.split("#")[0]
						if tmp_url[0:4] == "http":
							new_pages.append(tmp_url)
		pages = new_pages
	print("Finish {}".format(school_abbr))

if __name__ == "__main__":
	start_time = time.time()
	pool = multiprocessing.Pool()
	for i in range(len(official_url)):
		url = official_url[i]
		if url == None:
			continue
		# if url == None or i < 10 or i > 20:
		# 	continue
		print("Downloading {}...".format(url))
		pool.apply_async(crawl,args=([url],url.split(".")[1],str(task[i])+"-"+school_name[i]))

	pool.close()
	pool.join()

	end_time = time.time()
	print("Time: {}s".format(end_time - start_time))

# https://gaokao.chsi.com.cn/sch/search--ss-on,option-qg,searchType-1.dhtml
# https://zhuanlan.zhihu.com/Ehco-python

	# br = webdriver.Chrome()
	# br.get(url)
	# save_me = ActionChains(br).key_down(Keys.CONTROL).key_down('s').key_up('s').key_up(Keys.CONTROL)
	# save_me.perform()
	# print("done save")
	# browser.implicitly_wait(5)
	# enter = ActionChains(br).key_down(Keys.ENTER).key_up(Keys.ENTER)
	# enter.perform()