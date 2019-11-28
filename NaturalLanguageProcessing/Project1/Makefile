all: prerequirement data ngram lstm

prerequirement:
	pip install -r requirement.txt

data:
	python spider.py
	python cut.py

ngram:
	python ngram.py --N 2 --K 1
	python ngram.py --N 2 --K 3
	python ngram.py --N 2 --K 5
	python ngram.py --N 3 --K 1
	python ngram.py --N 3 --K 3
	python ngram.py --N 3 --K 5
	python ngram.py --N 4 --K 1
	python ngram.py --N 4 --K 3
	python ngram.py --N 4 --K 5

lstm:
	python lstm.py