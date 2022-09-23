# This module contains some useful CS recovery algorithms
import numpy as np

## Orthogonal Matching Pursuit (OMP)
def OMP(A: np.ndarray, y: np.ndarray, s: float, display = 'off'):
    '''
    A: Sampling matrix <numpy.ndarray>
       A has M rows with M being # of samples
       A has N columns with N being signal dimension
    y: Vector of Samples <numpy.ndarray>
       y has 1 row and M columns
    s: fraction of non-zero entries <float>
       0 < s < 1 
    display: 'on' prints the recovery info
             'off' switches the prints off <default>
    '''
    # First we find the signal dimension out of A
    M, N = A.shape
    
    # Check if the dimensions match
    if (y.shape[0] == M) and (y.shape[1] == 1):
        y = y.T
    elif (y.shape[0] != 1) or (y.shape[1] != M):
        print('Oops! Dimensions do not match.')
        return
    
    # Save samples in a column vector
    y = y.T
    
    # Calculate the sparsity of the signal
    spr = int(s * N) + 10 # add few extra to be on the safe side
    spr = min(spr, N) # make sure we are not exceeding N -- only happens in low-dimensional toy-examples
    
    # Initialization
    itr = 0 # iteration 0
    Supp = [] # empty support 
    r = y # initial residual
    Delta = np.linalg.norm(r, ord = 2) # norm of the residual
    
    # Now, we iterate
    while (itr <= spr) and Delta > 1e-8:
        # Calculate the residual
        r_samp = np.dot(A.T,r)
        
        # Calculate the index
        ind = np.argmax(np.abs(r_samp))
        Supp.append(ind)
        
        # LS recovery
        A_S = A[:, Supp]
        A_S_pin = np.linalg.pinv(A_S)
        z = np.dot(A_S_pin,y)
        
        # Construct the new sparse signal
        x = np.zeros((N,1)) # vector of all zeros
        np.put(x, np.array(Supp), z)
        
        # Stopping Criteria
        r = y - np.dot(A, x)
        Delta = np.linalg.norm(r, ord = 2)
        
        # We got for the next iteration
        itr += 1
    
    if display == 'on':
        if Delta <= 1e-8:
            print(f'It took {itr} iterations to find the signal')
        else:
            print(f"It didn't seem to converge. Recovery is not trustable. Increase the sparsity factor 's'")
            
    elif display != 'off':
        print("Wrong display, set it either 'off' or 'on'. It is now 'off'")
    
    
    return x