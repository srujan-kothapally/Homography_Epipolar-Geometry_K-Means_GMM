from scipy.stats import multivariate_normal as mvn
import numpy as np 
import cv2 
arr = np.array([[5.9,3.2],[4.6,2.9],[6.2, 2.8],[4.7 ,3.2],[5.5 ,4.2],[5.0, 3.0],[4.9 ,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]], np.float32)
mu1 = [6.2, 3.2]
mu2 = [6.6, 3.7]
mu3 = [6.5, 3.0]

sig1 = [[0.5+0.1, 0],[0, 0.5+0.1]]

k=3
nrow,ncol = arr.shape
mean_arr = np.asmatrix([mu1,mu2,mu3])
sigma_arr = np.array([np.asmatrix(sig1) for i in range(k)])
prior = np.ones(k)/k
post = np.asmatrix(np.empty((nrow, k), dtype=float))
data = arr

 

def e_step(k,mean_arr,sigma_arr,prior,post,data):
    nrow,ncol = data.shape
    for i in range(nrow):
        
        den = 0
        for j in range(k):
               num = (mvn.pdf(data[i, :], mean_arr[j].A1, sigma_arr[j]))* prior[j]
               den += num
               post[i, j] = num
        post[i, :] /= den
#        assert post[i, :].sum() - 1 < 1e-4

    return post
    
def m_step(k,mean_arr,sigma_arr,prior,post,data):
    nrow, ncol = data.shape
    for j in range(k):
        const = post[:, j].sum()
        prior[j] = 1/nrow * const
        _mu_j = np.zeros(ncol)
        _sigma_j = np.zeros((ncol, ncol))
        for i in range(nrow):
            _mu_j += (data[i, :] * post[i, j])
            _sigma_j += post[i, j] * ((data[i, :] - mean_arr[j, :]).T * (data[i, :] - mean_arr[j, :]))
                #print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
        mean_arr[j] = _mu_j / const
        sigma_arr[j] = _sigma_j / const
    print(mean_arr)
    return sigma_arr,mean_arr,prior

def fit(k,mean_arr,sigma_arr,prior,post,data):
        tol=1e-1
#        self._init()
        num_iters = 0
        ll = 1
        previous_ll = 0
        while(ll-previous_ll > tol):
            previous_ll = loglikelihood(k,mean_arr,sigma_arr,prior,post,data)
            post,sigma_arr,mean_arr,prior = _fit(k,mean_arr,sigma_arr,prior,post,data)
            num_iters += 1
            ll = loglikelihood(k,mean_arr,sigma_arr,prior,post,data)
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
    
def loglikelihood(k,mean_arr,sigma_arr,prior,post,data):
        
        ll = 0
        for i in range(nrow):
            tmp = 0
            for j in range(k):
                #print(self.sigma_arr[j])
                tmp += mvn.pdf(data[i, :],mean_arr[j, :].A1,sigma_arr[j, :])*prior[j]
            ll += np.log(tmp) 
        return ll  
def _fit(k,mean_arr,sigma_arr,prior,post,data):
        post = e_step(k,mean_arr,sigma_arr,prior,post,data)
        sigma_arr,mean_arr,prior = m_step(k,mean_arr,sigma_arr,prior,post,data)
        return post,sigma_arr,mean_arr,prior

i_sigma_arr = np.array([np.asmatrix(np.identity(ncol)) for i in range(k)])
 
i_mean_arr  =np.asmatrix(np.random.random((k, ncol)))

x=0
fit(k,mean_arr,sigma_arr,prior,post,arr)
   
#print(sigma_arr)
#print(mean_arr)