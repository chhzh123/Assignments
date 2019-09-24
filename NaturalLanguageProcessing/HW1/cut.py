import time
import pkuseg
import jieba
import thulac

with open("input.txt","r") as infile:
	intext = infile.read()
	# pkuseg
	print("Begin pkuseg...")
	seg = pkuseg.pkuseg()   # 以默认配置加载模型
	start = time.time()
	text = seg.cut(intext)  # 进行分词
	end = time.time()
	res = "|".join(text)
	outfile = open("output_pkuseg.txt","w")
	outfile.write(res)
	print("Time (pkuseg): {:.4f}s".format(end-start),end="\n\n")
	# pkuseg.test('input.txt', 'output.txt', model_name="news", nthread=20)

	# jieba
	print("Begin jieba...")
	start = time.time()
	text = jieba.cut(intext)  # 进行分词
	end = time.time()
	res = "|".join(text)
	outfile = open("output_jieba.txt","w")
	outfile.write(res)
	print("Time (jieba): {:.4f}s".format(end-start),end="\n\n")

	# thulac
	print("Begin thulac...")
	thu = thulac.thulac(seg_only=True)
	start = time.time()
	text = thu.cut(intext)  # 进行分词
	end = time.time()
	text = [t[0] for t in text]
	res = "|".join(text)
	outfile = open("output_thulac.txt","w")
	outfile.write(res)
	print("Time (thulac): {:.4f}s".format(end-start),end="\n\n")