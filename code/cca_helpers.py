import numpy as np
import scipy.stats as sts 
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from statsmodels.multivariate.cancorr import CanCorr
#import decoding as dec

def angle_basis(basis1,basis2):
    """ Subspace aligment calculation.

    This function recieves two orthogonal basis and compute the angle between them.


    Typical usage example:

    """

    basis1 -= np.mean(basis1,1)[:,None]
    basis2 -= np.mean(basis2,1)[:,None]


    u,s,vt = np.linalg.svd(basis2 @ basis1.T)
    return np.degrees(np.arccos(s[0]))

def get_resids(X,trials,labels):

    trials["idx"] = np.arange(len(trials),dtype=int)

    d_temp = np.zeros_like(X)

    for _,idx in trials.groupby(labels).idx:
        d_temp[idx] = X[idx] - np.mean(X[idx],0)[None,:]

    return d_temp

def align(X1, X2, m, cv=False):

    """
    aligns activity of two areas with CCA

    Parameters
    ----------
    X1 : trials x N
        first area
    X2 :  trials x N
        second area

    Returns
    -------

    proj1.T @ cdims1, proj2.T @ cdims2, cc.cancorr
    -> axes in neuron space area 1 and 2,canonical correlation

    """

    # change trial order so the 50/50 train-test split is random
    #n_trials = X1.shape[0]
    #idx = np.arange(n_trials)
    #np.random.shuffle(idx)
    #X1 = X1[idx]
    #X2 = X2[idx]

    if cv:
        X1_train,X1_test = X1[n_trials//2:], X1[:n_trials//2]
        X2_train,X2_test = X2[n_trials//2:], X2[:n_trials//2]
    else:
        X1_train,X1_test = X1,X1
        X2_train,X2_test = X2,X2
    

    # reduce dimensionality -> denoising
    # proj1 -> maping between raw and latent space 
    pca1 = PCA(n_components=m).fit(X1_train)
    pca2 = PCA(n_components=m).fit(X2_train)

    proj1 = pca1.components_[:m]
    X1_low_train, X1_low_test = X1_train @ proj1.T, X1_test @ proj1.T,

    proj2 = pca2.components_[:m]
    X2_low_train, X2_low_test = X2_train @ proj2.T,X2_test @ proj2.T

    # Find canonical dimensions on train data
    cc = CanCorr(X1_low_train, X2_low_train)
    cdims1 = cc.y_cancoef
    cdims2 = cc.x_cancoef

    # (test) projections on the canonical dimensions
    X1_proj = X1_low_test @ cdims1
    X2_proj = X2_low_test @ cdims2

    # compute canonical correlations
    ccs =  [sts.pearsonr(X1_proj[:,c],X2_proj[:,c])[0] for c in range(m)]
    
    #print(X1_proj.shape, X2_proj.shape)
    
    return proj1.T @ cdims1, proj2.T @ cdims2, X1_proj, X2_proj, ccs #cc.cancorr


def correct_sign(ref,axes):
    axes = np.array(axes)
    flip = []
    # first fold (0), neurons (:), first CC (0)
    for f in range(len(axes)):
        fold_n = axes[f][:,0].T
        flip.append(sts.pearsonr(ref,fold_n)[0])
    
    axes *= np.array(np.sign(flip))[:,None,None]

    return axes, flip


 
def sparseCCA(X,Y,numCC):


    '''Calculating regularized CCA, np.maximizing (X.wxMat)' . (Y.wyMat)
    Inputs X= n x p1   Y=n x p2
    reg -> type of regularization 1 : L1  and 2 : L2 
    L1 regularization parameters  cx > |wx|  and cy > |wy|  (for L2 cx and cy are both 1)
    numCC -> the Number of CCA Modes
    Output wxMat=p1 x numCC  , wyMat=p2 x numCC, rVec=numCC x 1 -> the correlation coefficient at each mode
    This code was developed based the the algorithm introduced by D. Witten and R. Tibshirani in the following paper:
    Witten, D. M. & Tibshirani, R. J. Extensions of sparse canonical correlation analysis with
    applications to genomic data. Stat Appl Genet Mol Biol 8, Article28, doi:10.2202/1544-6115.1470 (2009).'''

    tol=1e-5

    dx=np.shape(X)
    dy=np.shape(Y)

    wxMat=np.zeros([dx[1],numCC])
    wyMat=np.zeros([dy[1],numCC])
    rVec=np.zeros(numCC)

    X=X-np.mean(X,0)[None]
    X=X/np.std(X,0)[None]

    Y=Y-np.mean(Y,0)[None]
    Y=Y/np.std(Y,0)[None]

    X[np.isnan(X)]=0
    Y[np.isnan(Y)]=0

    for icc in range(numCC):

        wx=np.random.randn(dx[1])
        wy=np.random.randn(dy[1])

        wx=wx/np.linalg.norm(wx)
        wy=wy/np.linalg.norm(wy)

        wxp=np.zeros(dx[1])
        wyp=np.zeros(dy[1])

        iter=0
        while (np.linalg.norm(wx-wxp)/np.linalg.norm(wx) > tol) & (np.linalg.norm(wy-wyp)/np.linalg.norm(wy) > tol):
        
            wxp=wx
            wyp=wy

            wx = X.T @ Y @ wy
            wx=wx/np.linalg.norm(wx)

            wy=(wx.T @ X.T @ Y).T
            wy[np.isnan(wy)]=0
            wy=wy/np.linalg.norm(wy)

            iter=iter+1
            if iter>5000:
                break

        r=sts.pearsonr(X @ wx , Y @ wy)[0]

        wxMat[:,icc]=wx
        wyMat[:,icc]=wy
        rVec[icc]=r

        X=X-(X @ wx[:,None] @ wx[None,:])
        Y=Y-(Y @ wy[:,None] @ wy[None,:])

    return wxMat,wyMat,rVec
