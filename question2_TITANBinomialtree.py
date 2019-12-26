# -*- coding: utf-8 -*-
# @Author: shayaan
# @Date:   2019-12-04 21:31:35
# @Last Modified by:   shayaan
# @Last Modified time: 2019-12-26 10:56:56
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches



class TitanBinomialTreeOption(object):
	"""CRR model implementation """
	def __init__(self,option_type,security_price,strike_price,maturity,rate,vol,depth_of_tree):
		super(TitanBinomialTreeOption, self).__init__()
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
		
		self.v = math.exp(self.vol**2 * self.deltaT)
		self.u = 0.5 * math.exp(self.rate * self.deltaT) * self.v*(self.v+1 + math.sqrt(self.v**2 + 2*self.v -3) )
		self.d =  0.5 * math.exp(self.rate * self.deltaT) * self.v*(self.v+1 - math.sqrt(self.v**2 + 2*self.v -3) )
		self.p = (math.exp(self.rate*self.deltaT) - self.d) / (self.u - self.d)
		
		if (self.type == 'American Put'):
			self.EPA = [[0 for j in range(self.n+1)] for i in range(self.n+1)]
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
				if self.tree_option[self.n][i] != 0:
						self.EPA[self.n][i] = 1


	#Function to recursively compute the price of the option
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
					#Finding the exercise frontier
					if self.tree_option[i][j] == (self.strike_price-self.tree_security[i][j]):
						self.EPA[i][j] = 1

	def exerciseRegion(self):
		if self.type != 'American Put':
			print "Cannot plot exercise region"
			return
		# i = np.arange(self.n+1)
		data = np.array(self.tree_security)
		markers = ['r','b']
		classes = ["Early exercise",'No early exercise']
		for i in range(self.n+1):
			for j in range(i+1):
				if self.EPA[i][j] == 1:	
					a = plt.scatter(i,data[i][j],c=markers[0],marker='o')
				else:
					b = plt.scatter(i,data[i][j],c=markers[1],marker='^')
		
		recs = []
		for i in range(0,len(markers)):
			recs.append(mpatches.Rectangle((0,0),1,1,fc=markers[i]))
		plt.legend(recs,classes,loc=0)
		plt.xlabel('Number of steps')
		plt.ylabel('Price of the underlying security')
		plt.show()

if __name__ == '__main__':
	#Input format security price, strike price, time, rate, volatility, depth of the tree
	test = TitanBinomialTreeOption('American Put',100,100,2,0.05,0.2,10)
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


	if (test.type == 'American Put'):
		for i in range(test.n+1):
			for j in range(test.n+1):
				print (test.EPA[i][j]),
			print
		print 

		test.exerciseRegion()


	print("Price of the option {:0.2f}".format(test.tree_option[0][0]))