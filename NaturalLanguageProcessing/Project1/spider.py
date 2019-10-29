import sys, os, time
import requests
import bs4
from bs4 import BeautifulSoup
import multiprocessing

send_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36",
            "Connection": "keep-alive",
            "Accept-Language": "zh-CN,zh;q=0.8"
            }

url_file_name = "url.txt"
urls = []

if not os.path.isfile("url.txt"):
# if True:
	for section in ["it_2016","gd2016"]:
		for i in range(1,21):
			index_url = 'http://tech.163.com/special/' + section + ("/" if i == 1 else "_{:>02d}/".format(i))
			res = requests.get(index_url,timeout=30,headers=send_headers)
			soup = BeautifulSoup(res.content, 'lxml')
			titles = soup.find_all("h3",class_="bigsize ")
			print(index_url,len(titles))
			for news in titles:
				url = news.a["href"]
				if url.find("https://tech.163.com/") == -1:
					continue
				link = [url,news.a.contents[0]]
				if not link in urls:
					urls.append(link)
			print("Finish {}".format(i),flush=True)
	for i in range(1,11):
		index_url = 'http://tech.163.com/special/0009rt/ycbd' + (".html" if i == 1 else ("_0" + str(i) + ".html") if i != 10 else "_10.html")
		res = requests.get(index_url,timeout=30,headers=send_headers)
		soup = BeautifulSoup(res.content, 'lxml')
		titles = soup.find_all("span",class_="text")
		print(index_url,len(titles))
		for (j,news) in enumerate(titles):
			if j >= 100:
				break
			for news in titles:
				url = news.a["href"]
				if url.find("https://tech.163.com/") == -1:
					continue
				link = [url,news.a.contents[0]]
				if not link in urls:
					urls.append(link)
		print("Finish {}".format(i),flush=True)
	with open(url_file_name,"w",encoding="utf-8") as file:
		for url in urls:
			file.write(url[0] + "," + url[1] + "\n")
	print("Finish writing url file!")

else:
	url_file = open(url_file_name,"r",encoding="utf-8")
	for line in url_file:
		urls.append(line.split(","))

news_path = "news"

def crawl(url,outfile_name):
	html = requests.get(url,timeout=30,headers=send_headers)
	soup = BeautifulSoup(html.content,"lxml")
	title = soup.find("title").contents
	text = soup.find("div",class_="post_text")
	if text == None:
		soup = BeautifulSoup(html.content,"html.parser",from_encoding="gb18030")
		title = soup.find("title").contents
		text = soup.find("div",class_="post_text")
	try:
		old_title = text.find("p",class_="otitle").contents
	except:
		old_title = [""]
	main_text = text.find_all("p",class_=None)
	all_text = title + old_title + ["" if len(ct_with_tag.contents) == 0 or type(ct_with_tag.contents[0]) == bs4.element.Tag or ct_with_tag.contents[0] in ["StartFragment","EndFragment"] else ct_with_tag.contents[0] for ct_with_tag in main_text]
	outfile = open(outfile_name,"w",encoding="utf-8")
	outfile.write("\n\n".join(all_text))
	print("Finish {}".format(outfile_name),flush=True)

# if __name__ == "__main__":
# 	start_time = time.time()

# 	pool = multiprocessing.Pool()
# 	for (i,(url,title)) in enumerate(urls,1):
# 		if i > 1000:
# 			break
# 		outfile_name = news_path + "\\" + str(i) + ".txt"
# 		# if os.path.isfile(outfile_name):
# 		# 	continue
# 		pool.apply_async(crawl,args=(url,outfile_name))

# 	pool.close()
# 	pool.join()

# 	end_time = time.time()
# 	print("Time: {}s".format(end_time - start_time))

# 	not_done = []
# 	for (i,(url,title)) in enumerate(urls,1):
# 		if i > 1000:
# 			break
# 		outfile_name = news_path + "\\" + str(i) + ".txt"
# 		if os.path.isfile(outfile_name):
# 			continue
# 		not_done.append(i)
# 	print(not_done)

# crawl(url=urls[1][0],outfile_name=news_path + "\\" + str(2) + ".txt")

for (i,(url,title)) in enumerate(urls,1):
	if i > 1000:
		break
	infile_name = news_path + "\\" + str(i) + ".txt"
	if os.path.isfile(infile_name):
		infile = open(infile_name,"r",encoding="utf-8")
		outfile = open("news_new" + "\\" + str(i) + ".txt","w",encoding="utf-8")
		for line in infile:
			newtext = line
			if "_网易科技" in line:
				newtext = newtext.replace("_网易科技","")
			newtext = newtext.lstrip()
			newtext = newtext.rstrip()
			outfile.write(newtext + "\n")
	if i % 100 == 0:
		print("Finish {}/1000".format(i))