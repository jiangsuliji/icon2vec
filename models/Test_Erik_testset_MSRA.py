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
		
		
	def __call__(self, phraseText):
		"""given phraseText string, provide top N icons with scores"""
		print(self.msra(phraseText))
		
		
# # a simple example to call it
M = Tester()
M("man is good")




