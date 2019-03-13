# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from scipy.sparse import lil_matrix, csr_matrix
import pywt

class ADMM() :
    def __init__(self,
                 tau_rate  = 1,
                 beta_rate = 1,
                 mu_rate   = 1, 
                 rho       = 1.3,
                 tol       = 1e-6,
                 disp_iter = 10,
                 max_iter  = 1000,
                 max_time  = 600  ):
        self.tau_rate  = tau_rate
        self.beta_rate = beta_rate
        self.mu_rate   = mu_rate
        self.rho       = rho
        self.tol       = tol
        self.disp_iter = disp_iter
        self.max_iter  = max_iter
        self.max_time  = max_time
        
        
    def decomp_via_3mat(self, Data, m, n, r):
        #set parameters
        tau      = self.tau_rate  * max(m*n, r) ** (-0.5)
        beta1    = self.beta_rate * max(m*n, r) ** (-0.5)
        beta2    = self.beta_rate * max(m*n, r) ** (-0.5)
        beta3    = self.beta_rate * max(m*n, r) ** (-0.5)
        mu       = self.mu_rate   * max(m*n, r) ** (-0.5)
        max_mu   = mu        * 1e7
        rho      = self.rho
        tol      = self.tol
        disp_iter = self.disp_iter
        max_iter  = self.max_iter
        max_time  = self.max_time
        
        def _block_dif1(x):
            row, col = x.shape
            D = np.zeros((row-1, col))
            for j in range(col):
                for i in range(row-2):
                    D[i, j] = x[i, j] - x[i+1, j]
            return D
        
        def _block_dif2(x):
            row, col = x.shape
            D = np.zeros((row+1, col))
            for j in range(col):
                D[0, j] = x[0,j]
                for i in range(1, row):
                    D[i, j] = x[i, j] - x[i-1, j]
                D[row, j] = -x[row-1,j]
            return D
        
        def _mat_reshape_FtoW(X, m, n, r):
            ret = np.empty((m, n*r))
            for i in range(r):
                ret[:, i*n: (i+1)*n] = np.transpose(np.reshape(X[i, :], (n, m)))
            return ret
                
        def _mat_reshape_FtoH(X, m, n, r):
            ret = np.empty((n, m*r))
            for i in range(r):
                ret[:, i*m: (i+1)*m] = np.reshape(X[i, :], (n, m))
            return ret
        
        def _mat_reshape_WtoF(X, m, n, r):
            ret = np.empty((r, m*n))
            for i in range(r):
                ret[i, :] = np.reshape(np.transpose(X[:, i*n:(i+1)*n]), (1,m*n))
            return ret
        
        def _mat_reshape_HtoF(X, m, n, r):
            ret = np.empty((r, m*n))
            for i in range(r):
                ret[i, :] = np.reshape(X[:, i*m:(i+1)*m], (1,m*n))
            return ret
        
        constNormD = np.linalg.norm(Data, "fro")
        
        # create coefficient matrix
        LU1 = self._create_coefficient_lu_factor(r)
        LU2 = self._create_coefficient_lu_factor(m)
        LU3 = self._create_coefficient_lu_factor(n)
        
        # create index for transform
        '''
        indexF2H = np.zeros((m*n*r, 1))
        for i in range(m):
            for j in range(n*r):
                indexF2H[i + m * j]
        indexF2W = _create_index2(m,n,r)
        '''
        
        # initialize
        L = np.zeros((r, m*n))
        S = np.zeros((r, m*n))
        Q1 = np.zeros((r-1, m*n))
        Q2 = np.zeros((m-1, n*r))
        Q3 = np.zeros((n-1, m*r))
        Y1 = np.zeros((r, m*n))
        Y2 = np.zeros((r, m*n))
        Y3 = np.zeros((r, m*n))
        Y4 = np.zeros((r, m*n))
        Y5 = np.zeros((r-1, m*n))
        Y6 = np.zeros((m-1, n*r))
        Y7 = np.zeros((n-1, m*r))
        
        #ここから
        
        # loop
        for i in range(max_iter):
        #for i in range(10):
            
            # step 1: update L & J
            L = self._singular_threshold(Data - S + (Y1 / mu), 1/mu)
            J1 = scipy.linalg.lu_solve(LU1, S - (Y2 / mu) + _block_dif2(Q1 - (Y5 / mu)))
            J2 = scipy.linalg.lu_solve(LU2, _mat_reshape_FtoW(S - (Y3 / mu), m, n, r) + _block_dif2(Q2 - (Y6 / mu)))
            J3 = scipy.linalg.lu_solve(LU3, _mat_reshape_FtoH(S - (Y4 / mu), m, n, r) + _block_dif2(Q3 - (Y7 / mu)))
            AJ1 = _block_dif1(J1)
            AJ2 = _block_dif1(J2)
            AJ3 = _block_dif1(J3)
            
            J2_reshaped = _mat_reshape_WtoF(J2, m, n, r)
            J3_reshaped = _mat_reshape_HtoF(J3, m, n, r)
            
            # step 2: update S & Q
            S = pywt.threshold(0.25 * (Data - L + J1 + J2_reshaped + J3_reshaped + ((Y1 + Y2 + Y3 + Y4) / mu)), 0.25 * tau / mu, 'soft')
            Q1 = pywt.threshold(AJ1 + (Y5 / mu), beta1 / mu, 'soft')
            Q2 = pywt.threshold(AJ2 + (Y6 / mu), beta2 / mu, 'soft')
            Q3 = pywt.threshold(AJ3 + (Y7 / mu), beta3 / mu, 'soft')

            # step 3: update Lagrangian multipliers
            Z  = L + S - Data
            Y1 = Y1 - mu*Z
            Y2 = Y2 - mu*(S - J1)
            Y3 = Y3 - mu*(S - J2_reshaped)
            Y4 = Y4 - mu*(S - J3_reshaped)
            Y5 = Y5 - mu*(Q1 - AJ1)
            Y6 = Y6 - mu*(Q2 - AJ2)
            Y7 = Y7 - mu*(Q3 - AJ3)
            mu = min(max_mu, rho*mu)
            
            # step 4: termination
            normD = np.linalg.norm(Z, "fro") / constNormD
                
            ## display
            if np.mod(i, disp_iter) == 0:
                print('iter:' + format(i+1, '03d')
                        + '  normD:' + format(normD, '.5e')
                        + '  rank(L):' + format(np.linalg.matrix_rank(L), '03d')
                        + '  nonzero(S):' + format(np.count_nonzero(S), 'd'))
                
            if normD < tol:
                break;
                
        return L, S
    
    
    def decomp_via_vector(self, Data, m, n, r):
        #set parameters
        tau      = self.tau_rate  * max(m*n, r) ** (-0.5)
        beta     = self.beta_rate * max(m*n, r) ** (-0.5)
        mu       = self.mu_rate   * max(m*n, r) ** (-0.5)
        max_mu   = mu        * 1e7
        rho      = self.rho
        tol      = self.tol
        disp_iter = self.disp_iter
        max_iter  = self.max_iter
        max_time  = self.max_time
        
        def _block_dif1( x, m, n, r):
            l = m*n*r
            r = np.zeros((3*l, 1))
            for i in range(l):
                r[    i] = x[i] - x[np.mod(i+1  , l)]
                r[  l+i] = x[i] - x[np.mod(i+m  , l)]
                r[2*l+i] = x[i] - x[np.mod(i+m*n, l)]
            return r
        
        def _block_dif2( x, m, n, r):
            l = m*n*r
            r1 = np.zeros((l, 1))
            r2 = np.zeros((l, 1))
            r3 = np.zeros((l, 1))
            for i in range(l):
                r1[i] = x[      i] - x[      np.mod(i-1  , l)]
                r2[i] = x[  l + i] - x[  l + np.mod(i-m  , l)]
                r3[i] = x[2*l + i] - x[2*l + np.mod(i-m*n, l)]
            return r1+r2+r3
        
        
        constNormD = np.linalg.norm(Data, "fro")
        
        # create coefficient vector
        denominator = self._create_resource_vector(m,n,r)
        
        # initialize
        L = np.zeros((r, m*n))
        S = np.zeros((r, m*n))
        Q = np.zeros((3*m*n*r, 1))
        Y1 = np.zeros((r, m*n))
        Y2 = np.zeros((r, m*n))
        Y3 = np.zeros((3*m*n*r, 1))
        
        # loop
        for i in range(max_iter):
        #for i in range(10):
            
            # step 1: update L & J
            L = self._singular_threshold(Data - S + (Y1 / mu), 1/mu)
            vecJ = self._updateJ(np.reshape(S - (Y2 / mu), (m*n*r, 1)) + _block_dif2(Q - (Y3 / mu), m, n, r), denominator)
            J = np.reshape(vecJ, (r, m*n))
            AJ = _block_dif1(vecJ, m,n,r)
            
            # step 2: update S & Q
            S = pywt.threshold(0.5 * (Data - L + (Y1 / mu) + J + (Y2 /mu)), 0.5 * tau / mu, 'soft')
            Q = pywt.threshold(AJ + (Y3 / mu), beta / mu, 'soft')

            # step 3: update Lagrangian multipliers
            Z  = L + S - Data
            Y1 = Y1 - mu*Z
            Y2 = Y2 - mu*(S-J)
            Y3 = Y3 - mu*(Q-AJ)
            mu = min(max_mu, rho*mu)
            
            # step 4: termination
            normD = np.linalg.norm(Z, "fro") / constNormD
                
            ## display
            if np.mod(i, disp_iter) == 0:
                print('iter:' + format(i+1, '03d')
                        + '  normD:' + format(normD, '.5e')
                        + '  rank(L):' + format(np.linalg.matrix_rank(L), '03d')
                        + '  nonzero(S):' + format(np.count_nonzero(S), 'd'))
                
            if normD < tol:
                break;
                
        return L, S
        
        
    def _create_resource_vector(self, m, n, r):
        # create index vector
        p = m*n*r
        d = np.zeros((p,1))
        d[0    ] = 7
        d[1    ] = -1
        d[p-1  ] = -1
        d[m    ] = -1
        d[p-m  ] = -1
        d[m*n  ] = -1
        d[p-m*n] = -1
   
        # Fourier Transform
        d = np.fft.fftn(d)
        return d.real
    
    def _create_coefficient_lu_factor(self, N):
        #A = lil_matrix(N-1, N)
        A = np.zeros((N-1, N))
            
        for i in range(N-1):
            A[i,i]   =  1
            A[i,i+1] = -1
                
        #A = A.tocsr()
        
        LU = scipy.linalg.lu_factor(np.eye(N) + np.dot(A.T, A))
        
        return LU
    
    
    def _singular_threshold(self, X, tsd):
        U, S, V = np.linalg.svd(X, full_matrices = False)
        return np.dot(np.dot(U, np.diag(pywt.threshold(S, tsd, 'soft'))), V)
    
    def _updateJ(self, b, denominator):
        c = np.fft.ifftn(np.fft.fftn(b) / denominator)
        return c.real
    
          
    
    
    
    
    
    
    