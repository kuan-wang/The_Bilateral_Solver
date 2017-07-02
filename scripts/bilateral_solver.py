# %pylab inline
# %matplotlib inline
# import pylab
from pylab import *
import seaborn as sns


import time
from datetime import datetime


sns.set_style('white')
sns.set_context('notebook')
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'CMRmap'
rcParams['figure.facecolor'] = 'w'



from skimage.io import imread
from skimage import transform
import os

# data_folder = os.path.abspath(os.path.join(os.path.curdir, '..', 'data', 'epfl_corridor', '20141008_141749_00'))
# data_folder = os.path.abspath(os.path.join(os.path.curdir, '..', 'data', 'epfl_lab', '20140804_160621_00'))
data_folder = os.path.abspath(os.path.join(os.path.curdir, '..', 'data', 'depth_superres'))
# print (data_folder)

# The RGB image that whose edges we will respect
# reference = imread(os.path.join(data_folder, 'rgb000120.png'))
reference = imread(os.path.join(data_folder, 'reference.png'))
# The 1D image whose values we would like to filter
# target = imread(os.path.join(data_folder, 'depth000120.png'))
target = imread(os.path.join(data_folder, 'target.png'))
# A confidence image, representing how much we trust the values in "target".
# Pixels with zero confidence are ignored.
# Confidence can be set to all (2^16-1)'s to effectively disable it.
confidence_0 = imread(os.path.join(data_folder, 'confidence.png'))

# confidence = confidence_0[:target.shape[0],:target.shape[1],:]
confidence = transform.resize(confidence_0, (target.shape[0], target.shape[1]))

im_shape = reference.shape[:2]
assert(im_shape[0] == target.shape[0])
assert(im_shape[1] == target.shape[1])
assert(im_shape[0] == confidence.shape[0])
assert(im_shape[1] == confidence.shape[1])


#figure(figsize=(14, 20))
subplot(311)
imshow(reference)
title('reference')
subplot(312)
imshow(confidence)
title('confidence')
subplot(313)
imshow(target)
title('target')




RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))




MAX_VAL = 255.0
from scipy.sparse import csr_matrix

def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=8, sigma_luma=4, sigma_chroma=4):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        print ("luma_coords:",luma_coords.shape)
        print ("chroma_coords:",chroma_coords.shape)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        print ("coords:",coords.shape)
        print ("coords_flat:",coords_flat.shape)
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # print (self.dim)
        # print (self.hash_vec.shape)
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True)
        print ("hashed_coords.shape:",hashed_coords.shape)
        print ("unique_hashes.shape:",unique_hashes.shape)
        print ("unique_idx.shape:",unique_idx.shape)
        print ("idx.shape:",idx.shape)
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # print ("nvertices:",self.nvertices)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # print ("idx:",idx)
        # print ("S.shape:",self.S)
        # print ("idx:",idx)
        # print ("S:",self.S)
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        # print ("hash_vec:",self.hash_vec)
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))



from scipy.sparse import diags
from scipy.sparse.linalg import cg

def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    start_ = datetime.utcnow()
    print ("start bistochastize:",start_)
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    print ("Dm.shape:",Dm.shape)
    print ("Dn.shape:",Dn.shape)
    end_ = datetime.utcnow()
    print ("end bistochastize:",end_)
    return Dn, Dm

class BilateralSolver(object):
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        start_ = datetime.utcnow()
        print ("start BilateralSolver.solve:",start_)
        if w.ndim == 2:
            assert(w.shape[1] == 1)
        elif w.dim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:,0], 0)
        A = self.params["lam"] * A_smooth + A_data
        print ("A.shape:",A.shape)
        xw = x * w
        b = self.grid.splat(xw)
        print ("b.shape:",b.shape)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        print ("A_diag.shape:",A_diag.shape)
        M = diags(1 / A_diag, 0)
        print ("1/A_diag.shape:",(1/A_diag).shape)
        print ("M.shape:",M.shape)
        # Flat initialization
        y0 = self.grid.splat(xw) / w_splat
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"], tol=self.params["cg_tol"])
        xhat = self.grid.slice(yhat)
        end_ = datetime.utcnow()
        print ("end BilateralSolver.solver:",end_)
        return xhat





grid_params = {
    'sigma_luma' : 4, # Brightness bandwidth
    'sigma_chroma': 4, # Color bandwidth
    'sigma_spatial': 8 # Spatial bandwidth
}

bs_params = {
    'lam': 128, # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5, # The tolerance on the convergence in PCG
    'cg_maxiter': 25 # The number of PCG iterations
}


start_ = datetime.utcnow()
print ("start creating BilateralGrid:",start_)
# %%time
grid = BilateralGrid(reference, **grid_params)
end_ = datetime.utcnow()
print ("end creating BilateralGrid:",end_)



# %%time
t = target.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)
c = confidence.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)

tc_filt = grid.filter(t * c)
c_filt = grid.filter(c)
output_filter = (tc_filt / c_filt).reshape(im_shape)

# %%time
output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)



imargs = dict(vmin=0, vmax=1)
figure(figsize=(14, 24))
# subplot(311)
# imshow(t.reshape(im_shape), **imargs)
# title('input')
# subplot(312)
# imshow(output_filter, **imargs)
# title('bilateral filter')
subplot(111)
imshow(output_solver, **imargs)
title('bilateral solver')
show()
