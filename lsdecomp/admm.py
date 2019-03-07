# -*- coding: utf-8 -*-

import numpy as np
import pywt

class ADMM() :
    def __init__(self, tau_rate = 1, beta_rate = 1, mu_rate = 1, \
                 rho = 1.3, tol = 1e-6, disp_iter = 10, max_iter = 1000, max_time = 600):
        self.tau_rate  = tau_rate
        self.beta_rate = beta_rate
        self.mu_rate   = mu_rate
        self.rho       = rho
        self.tol       = tol
        self.disp_iter = disp_iter
        self.max_iter  = max_iter
        self.max_time  = max_time
        
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
        
        
    #def decomp_via_3mat():
        
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
    
    def _singular_threshold(self, X, tsd):
        U, S, V = np.linalg.svd(X, full_matrices = False)
        return np.dot(np.dot(U, np.diag(pywt.threshold(S, tsd, 'soft'))), V)
    
    def _updateJ(self, b, denominator):
        c = np.fft.ifftn(np.fft.fftn(b) / denominator)
        return c.real
    
          
    
    
    
    
    
    
    