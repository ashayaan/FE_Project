import numpy as np 
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

###################################################################
################## DEFINING CONSTANTS #############################
###################################################################

K 				= 100.0
S_MAX 			= 200.0
T 				= 1
SIGMA 			= 0.2
r 				= 0.1
M 				= 50
N 				= 20
h 				= float(S_MAX)/float(M+1)
DELTA_T 		= float(T)/float(N) 
num_iterations 	= 1000

###################################################################
########## Part 1: Generating the Matrix A ########################
###################################################################

x = np.array([h*j for j in range(M+1)])
t = np.array([DELTA_T*j for j in range(N)])

A = [[0 for i in range(M+1)] for j in range(M+1)]

for i in range(M):
	
	if(i>0):
		A[i][i-1] = - (((SIGMA**2)*(x[i]**2))/(2*(h**2))) 
	
	A[i][i] =  (((SIGMA**2)*(x[i]**2))/((h**2))) + (r*x[i]/h) + r
	A[i][i+1] = - (((SIGMA**2)*(x[i]**2))/(2*(h**2))) - (r*x[i]/h)

A = np.array(A)

###################################################################
############ Part 3: Implementing the Explicit Scheme #############
################################################################### 

def phi_call(x):
	payoff = max(x-K,0)
	return payoff

def phi_put(x):
	payoff = max(K-x,0)
	return payoff

P = np.array([[0.0 for i in range(M+1)] for j in range(N+1)])

for i in range(M+1):
	P[0][i] = phi_call(x[i])


for i in range(1,N+1):
	P[i] = np.maximum(P[i-1]-DELTA_T*np.matmul(A,P[i-1]),np.array([phi_call(x[j]) for j in range(M+1)])) 

print("Explicit Euler Pricing Solution:")
print(P[N])
explicit_sol = P[N]

# plt.plot(x,explicit_sol,'bo',markersize=8)
# plt.xlabel('Underlying Price')
# plt.ylabel('Option Price')

# plt.show()

###################################################################
############ Part 4: Implementing the Implicit Scheme #############
###################################################################


B = np.identity(M+1) + DELTA_T*A

P = [np.array([np.random.uniform() for i in range(M+1)]) for i in range(N+1)]

for i in range(M+1):
	P[0][i] = phi_call(x[i])

def F(a,b):
	return np.minimum(np.matmul(B,a)-b,a-P[0])


def deriv(a,b,i,j):

	p = np.matmul(B,a)-b
	q = a - P[0]

	if(p[i]<=q[i]):
		return B[i][j]

	else:
		if(i==j):
			return 1
		else:
			return 0

def F_dash(a,b):

	K = [[0 for i in range(M+1)] for i in range(M+1)]
	
	for i in range(M+1):
		for j in range(M+1):
			K[i][j] = deriv(a,b,i,j)

	return np.array(K)


for j in range(1,N+1):
	for i in range(num_iterations):
		P[j] = P[j] - np.matmul(np.linalg.pinv(F_dash(P[j],P[j-1])),F(P[j],P[j-1]))

	print("Time Step Complete: "+ str(j))

print("Implicit Euler Pricing Solution: ")
implicit_sol = P[N]
plt.plot(x,implicit_sol,'bo',markersize=8)
plt.xlabel('Underlying Price')
plt.ylabel('Option Price')

plt.show()









