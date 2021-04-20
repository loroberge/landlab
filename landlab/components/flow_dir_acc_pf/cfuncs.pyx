import numpy as np
cimport numpy as np
cimport cython


DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t


cpdef _FD_D8(np.ndarray[DTYPE_INT_t, ndim=1] receivers,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] distance_receiver,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] steepest_slope,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] el,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] el_ori,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] dist,
             np.ndarray[DTYPE_INT_t, ndim=1] ngb,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] el_d,
             DTYPE_INT_t r, DTYPE_INT_t c,
             DTYPE_FLOAT_t dx, DTYPE_FLOAT_t sq2):
    
    """
    Calcualte D8 flow dirs
    """
    cdef int idx, i
    
    for i in range(0,r*c):      
        
        if receivers[i] != -1:  
            ngb[0] = i + 1
            ngb[1] = i + c 
            ngb[2] = i - 1
            ngb[3] = i -c
            ngb[4] = i + c + 1
            ngb[5] = i + c - 1 
            ngb[6] = i -c - 1
            ngb[7] = i -c + 1
                
            # Differences after filling can be very small, *1e3 to exaggerate those
            el_d[0] = (el[i] - el[i + 1])*1e3
            el_d[1] = (el[i] - el[i + c])*1e3
            el_d[2] = (el[i] - el[i -1])*1e3
            el_d[3] = (el[i] - el[i - c])*1e3
            el_d[4] = (el[i] - el[i + c + 1])*1e3/(sq2)
            el_d[5] = (el[i] - el[i + c - 1])*1e3/(sq2)
            el_d[6] = (el[i] - el[i -c - 1])*1e3/(sq2)
            el_d[7] = (el[i] - el[i -c + 1])*1e3/(sq2)            
         
                   
            idx = np.argmax(el_d)
            receivers[i] = ngb[idx]
            # Slope should be able to have negative values 
            distance_receiver[i]=dist[idx]
            steepest_slope[i]  = (el_ori[i]-el_ori[receivers[i]])/distance_receiver[i]
            
          
    



cpdef _FA_D8(DTYPE_INT_t np,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] a,
             np.ndarray[DTYPE_FLOAT_t, ndim=1] q,
             np.ndarray[DTYPE_INT_t, ndim=1] stack,
             np.ndarray[DTYPE_INT_t, ndim=1] receivers):
    """
    Accumulates drainage area and discharge, permitting transmission losses.
    """
    cdef int donor, recvr, i

    # Iterate backward through the list, which means we work from upstream to
    # downstream.
    for i in range (np-1,-1,-1):
            donor = stack[i]
            if receivers[donor] != -1:
                rcvr = receivers[donor]
                a[rcvr] += a[donor]
                q[rcvr] += q[donor]            
    
