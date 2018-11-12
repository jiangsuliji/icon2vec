"""Test MSRA's perf on Erik's testset"""

import msra
import requests
import os

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class Tester:
	"""compare ML and MSRA and (our model)"""
	def __init__(self):
		self.msra = msra.Model_MSRA()
		self.N = 2 #number of icons to be suggested
		self.loadData()
		
	def loadData(self):
		filePath = "..\data\\benchmarks\ErikOveson_11_05\\testset_SingleIcon_9-18_10-18-2018_025Unk_MinWord3_Kept24Hrs.ss.csv"
		self.data = []
		with open(filePath, "r", encoding="utf8") as f:
			idx = 0
			for line in f:
				if idx == 0:
					idx += 1
					continue
				items = line.split(',')
				if len(items) < 3:
					continue
				label = items[1][9:]
				phrase = ','.join(items[2:])[:-1]
				self.data.append([phrase, label])
		print("parsed ", len(self.data), "test entries. Sample entry:", self.data[0])
		
	def __call__(self, phraseText):
		"""given phraseText string, provide top N icons with scores"""
		return(self.msra(phraseText))
		
	def runTest(self):
		T, P = 0, 0
		idx = 0
		for phrase, label in self.data[:1000]:
			# phrase = phrase.encode("utf-8")
			print(idx)
			idx += 1
			res = self.msra(phrase)
			# print(phrase, label, res)
			if label in (res[0][0], res[1][0]):
				T += 1
			else:
				P += 1
		print(T, P, T+P, T/(T+P))
	
	
		
# # a simple example to call it
M = Tester()
# M("man is good")
M.runTest()




