import cvxpy as cp
import numpy as np 
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

'hankel generates a boolean matrix'
def Hankel(n,i):
    H = np.zeros((n, n), int)
    for j in range(0,n):
        for k in range (0,n):
            if j+k == i:
                H[j,k]=1
    return H

def SP(A, i):
    M3 =  np.zeros((len(A)+1,len(A)+1), int)
    M3[0,i+1] = 1  
    M3[i+1,0] = 1
    return(M3)

'generates the constants of the 3rd PSD matrix'
def ZMat(A,p):
    Z  = np.zeros((len(A)+1,len(A)+1))
    np.fill_diagonal(Z, 1) 
    Z[0,0]=0
    for i in range(0,len(A)):
        Z[0,i+1]=-p[i] 
        Z[i+1,0]=-p[i]
    return Z        

'generates both Hankel matrices'
def HankelOne(d,m):
    e = d//2
    if  d%2==1:
       
        L1 = []
        L2 = []
        
        for i in range(0, d):
            L1.append(Hankel(e+1,i)*m[i+1])
            L2.append(Hankel(e+1,i)*m[i])
        C1 = sum(L1)
        C2 = sum(L2)-C1 
        return(C1,C2)
    else:
       
        L1 = []
        L2 = []
        L3 = []

        for i in range (0,d):
            L1.append(Hankel(e+1,i)*m[i])
        for i in range (1,d):
            L2.append(Hankel(e,i-1)*m[i-1])
            L3.append(Hankel(e,i-1)*m[i])    
        C1 = sum(L1)
        C2 = sum(L2)-sum(L3)
        return(C1,C2)
    
def ThirdC(A,m):
    D = np.zeros((len(A)+1,len(A)+1))
    D[0,0]=1
    L3 = [D*l]
    for j in range(0,len(A)):
        L3.append(SP(A,j)*m[A[j]])
    return(sum(L3))

'LinC adds up all variables indexed in A'
def LinC(A,m):
    S = sum([m[i] for i in A])
    return(S)

'The input' 
###############
n =10
B = [ncr(i,2)-1 for i in range(2,n)]
###############


'The dimension of the problem is d+1,'
'the variables we used are stored in m and l'

d=max(B)
'm is the vector of variables, representing the point on the '
'coalescence manifold, before projection.'
m = cp.Variable(d+1)
'l is the square of the distance from p to m'
l = cp.Variable()

'We generate a random point p to compute its distance'
'to the coalescence manifold.'

p = np.random.rand(len(B))
print("The random point p:",p)

    
'Collect all the matrices to make the three PSD constraints'


C1,C2 = HankelOne(d,m)

C3 = ThirdC(B,m)+ZMat(B,p)

'The linear constraint $L==1$ ensure we take a slice of the cone,'
'all coordinates sum to 1'
L = LinC(B,m)

'Generate the semi definite program '

constraints = [C1>>0, C2>>0, C3>>0, L==1]

prob = cp.Problem(cp.Minimize(l),constraints)

'If the status of the problem is inaccurate, increase the # of iterations'
prob.solve(verbose=True,max_iters=5000)



print('The square of the distance to the cone:',prob.value)
print('The point on the cone:',m.value)



        
        

  
