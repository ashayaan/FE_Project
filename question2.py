# -*- coding: utf-8 -*-
# @Author: shayaan
# @Date:   2019-12-04 21:31:35
# @Last Modified by:   shayaan
# @Last Modified time: 2019-12-05 11:47:32
import math


class CRR(object):
	"""CRR model implementation """
	def __init__(self,option_type,security_price,strike_price,maturity,rate,vol,depth_of_tree):
		super(CRR, self).__init__()
		self.type = option_type
		self.n = depth_of_tree 
		self.security_price = security_price 
		self.strike_price = strike_price 
		self.maturity = maturity 
		self.rate = rate 
		self.vol = vol 
		self.tree_security = [[0 for j in range(self.n+1)] for i in range(self.n+1)]
		self.tree_option = [[0 for j in range(self.n+1)] for i in range(self.n+1)]
		self.deltaT  = float(self.maturity) / float(self.n)
		self.discount = math.exp(-self.rate*self.deltaT)
		self.u = math.exp(self.vol * math.sqrt(self.deltaT))
		self.d = math.exp(-self.vol* math.sqrt(self.deltaT))
		self.p = (math.exp(self.rate*self.deltaT) - self.d) / (self.u - self.d)

		self.buildTree()
		self.terminalPrices()

	#Building the price binary tree
	def buildTree(self):
		for i in range(self.n+1):
			for j in range(i+1):
				self.tree_security[i][j] = self.security_price * math.pow(self.u,j) * math.pow(self.d,i-j)

	#Beginning back propagation
	def terminalPrices(self):
		for i in range(self.n+1):
			if self.type == 'European Call':
				self.tree_option[self.n][i] = max(self.tree_security[self.n][i] - self.strike_price,0.0)
			elif self.type == 'European Put':
				self.tree_option[self.n][i] = max(self.strike_price - self.tree_security[self.n][i],0.0)
			elif self.type == 'American Call':
				self.tree_option[self.n][i] = max(self.tree_security[self.n][i] - self.strike_price,0.0)
			elif self.type == 'American Put':
				self.tree_option[self.n][i] = max(self.strike_price - self.tree_security[self.n][i],0.0)

	def computePrice(self):
		for i in range(self.n-1,-1,-1):
			for j in range(i+1):
				# print(i,j)
				if self.type == 'European Call':
					self.tree_option[i][j] = math.exp(-1*self.rate*self.deltaT)*((1-self.p) * self.tree_option[i+1][j] + (self.p)*self.tree_option[i+1][j+1])
				elif self.type == 'European Put':
					self.tree_option[i][j] = math.exp(-1*self.rate*self.deltaT)*((1-self.p) * self.tree_option[i+1][j] + (self.p)*self.tree_option[i+1][j+1])
				elif self.type == 'American Call':
					self.tree_option[i][j] = max(self.tree_security[i][j]-self.strike_price,math.exp(-1*self.rate*self.deltaT)*((1-self.p)*self.tree_option[i+1][j] + (self.p)*self.tree_option[i+1][j+1]) )
				elif self.type == 'American Put':
					self.tree_option[i][j] = max(self.strike_price-self.tree_security[i][j],math.exp(-1*self.rate*self.deltaT)*((1-self.p)*self.tree_option[i+1][j] + (self.p)*self.tree_option[i+1][j+1]) )


if __name__ == '__main__':
	test = CRR('European Call',100,100,2,0.05,0.2,2)
	test.computePrice()
	
	for i in range(test.n+1):
		for j in range(test.n+1):
			print ("{:0.2f}".format(test.tree_security[i][j])),
		print 

	print

	for i in range(test.n+1):
		for j in range(test.n+1):
			print ("{:0.2f}".format(test.tree_option[i][j])),
		print
	print 	
	print(test.tree_option[0][0])