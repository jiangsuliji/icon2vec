"""MSRA model caller"""

import common
import secret
import requests
import os

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class Model_MSRA:
	"""class for calling MSRA's model"""
	def __init__(self):
		self.mp_icondescription2filename = common.mp_icondescription2filename
		self.MSRA_query_template = common.MSRA_query_template
		self.URL = secret.MSRA_URL
		self.N = 2 #number of icons to be suggested
		
		
	def __call__(self, phraseText, phraseEmbedding=None):
		"""given phraseText string, provide top N icons with scores"""
		self.MSRA_query_template['keywords'] = phraseText
		r = requests.post(url = self.URL, data = self.MSRA_query_template)
		# print(r.status_code, r.reason, r.headers['content-type'])
		if r.status_code != 200:
			raise
		# print(r.text)
		self.dumpHTML2Txt(r.text)
		return self.processResponse()
		
		
	def processResponse(self):
		"""given the text response, find the top N suggested icons"""
		# sample return: [['Man.svg', 0.05994971841573715], ['MaleProfile.svg', 0.0498545840382576]]
		inf = open('tmp.txt', 'r', encoding="utf-8")
		TMPresults = [[0, 0] for i in range(self.N+1)]
		for idx, line in enumerate(inf):
			# print("working on:", idx, line)
			# parse name
			if idx >=57 and idx < 57+self.N+1:
				t = line.split()
				# print(t)
				if len(t)>=2:
					t = t[1].split(">")
					if len(t)>=2:
						t = t[1].split("<")
					# print(t[0])
						TMPresults[idx-57][0] = str(t[0])
				continue
			# parse score
			if idx >=79 and idx < 79+self.N+1:
				t = line.split()
				if len(t)>=2:
					t = t[1].split(">")
					if len(t)>=2:
						t = t[1].split("<")
					# print(t[0])
						TMPresults[idx-79][1] = float(t[0])
				continue
		TMPresults = TMPresults[1:]
		# print(TMPresults)
		inf.close()
		os.remove('tmp.txt')
		return TMPresults
		
		
	def dumpHTML2Txt(self, responseText):
		inf = open('tmp.txt', 'w', encoding="utf-8")
		inf.write(responseText)
		inf.close()
		
		
# # a simple example to call it
# M = Model_MSRA()
# res = M("man")
# print(res)



