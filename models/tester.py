"""Tester to test MSRA/ML model given a testset"""

import msra
import ml
import requests
import os

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

class Tester:
	"""compare ML and MSRA and (our model)"""
	def __init__(self):
		self.
		self.ml = ml.Model_ML()
		self.msra = msra.Model_MSRA()
		
		self.N = 2 #number of icons to be suggested
		
		
	def __call__(self, phraseText):
		"""given phraseText string, provide top N icons with scores"""
		print(self.ml(phraseText))
		print(self.msra(phraseText))
		
		
# # a simple example to call it
M = Tester()
M("man is good")




