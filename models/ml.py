"""ML TAS model caller"""

import numpy as np
import common
import secret
import requests
import json
import re

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class Model_ML:
	"""class for calling ML's model"""
	def __init__(self):
		self.mp_icondescription2filename = common.mp_icondescription2filename
		self.ML_query_template = common.ML_query_template
		self.header = secret.TAS_header
		self.URL = secret.TAS_URL
		self.N = 2 #number of icons to be suggested
		
		
	def __call__(self, phraseText, phraseEmbedding=None):
		"""given phraseText string, provide top N icons with scores"""
		self.ML_query_template["TextInputData"]["Title"] = phraseText
		r = requests.post(url = self.URL, json = self.ML_query_template, headers=self.header)
		# print(r.status_code, r.reason, r.headers['content-type'])
		if r.status_code != 200:
			raise
		# print(r.text)
		self.processResponse(r.text)
		
		
	def processResponse(self, responseText):
		"""given the TAS text response, find the top N suggested icons"""
		# sample return: [('Man.svg', 0.05994971841573715), ('MaleProfile.svg', 0.0498545840382576)]
		r = json.loads(responseText)['presentationAnalyses']['slideAnalyses'][0]['text2IconOutput']['iconInfos']['m_iconInfoList']
		print(r)
		if len(r) != self.N:
			raise
		res = [(self.mp_icondescription2filename[item['m_iconId']]+".svg", item['m_score']) for item in r]
		res = sorted(res, key=lambda k:k[1], reverse=True)
		# print(res)
		return res
		
		
		
m = Model_ML()
