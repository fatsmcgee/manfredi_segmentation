import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn import svm
import maxflow

def mask_from_image(image,imtype):
    if imtype == 'flowers':
        return np.logical_and(image[:,:,2]==128,image[:,:,1]==0)

#1. rimages,masks = load_images('flowers',100)
def load_images(imtype, n_images,rand_order,skip=0):
    
    rimages = []
    masks = []
    
    i_m_fs = None
    newsize = None
    
    if imtype == 'flowers':
        im_fs = os.listdir("../flower_images")
        m_fs = ['../flower_segments/' + f[:-4] + ".png"\
                            for f in im_fs]
        im_fs = ['../flower_images/' + f for f in im_fs]
        i_m_fs = [t for t in zip(im_fs,m_fs) if os.path.isfile(t[1])]
        
        newsize = (256,256)
        
    if rand_order:
        np.random.shuffle(i_m_fs)
        
    for im_f,m_f in i_m_fs[skip:skip+n_images]:
        im = cv2.imread(im_f)
        rimages.append(cv2.resize(im,newsize))
        
        m = cv2.imread(m_f)
        m = cv2.resize(m,newsize)
        masks.append(mask_from_image(m,imtype))
                
    im_fs = [t[0] for t in i_m_fs]
    m_fs = [t[1] for t in i_m_fs]
    return rimages,masks,im_fs,m_fs
        
def get_quantized_image(image,qbins):
    maxval = np.iinfo(image.dtype).max
    quantized = np.zeros((image.shape[0],image.shape[1]),dtype='uint32')
    for c in range(3):
        channel = qbins*(image[:,:,c]/float(maxval+1))
        channel = channel.astype('uint32')
        quantized[:,:] += channel * (qbins**c)
    return quantized
    
#2. qimages = get_quantized_images(rimages,qbins)
def get_quantized_images(rimages,qbins):
    qimages = [get_quantized_image(i,qbins) for i in rimages]
    return qimages
    
    
def get_hog_features(image):
    rows,cols,_ = image.shape
    
    #ala Manfredi
    cells_per_im = 5
    while rows%cells_per_im != 0:
        rows -=1
    while cols%cells_per_im !=0:
        cols -=1
    
    cell_length = min(rows/cells_per_im, cols/cells_per_im)
    pixels_per_cell = (cell_length,cell_length)
    cells_per_block = 3
    cells_per_stride = 2 #that is, overlap of one cell
    pixels_per_block = (cell_length * cells_per_block,cell_length*cells_per_block )
    pixels_per_stride = (cell_length * cells_per_stride, cell_length * cells_per_stride)
    bins = 9
    
    
    winSize = (cols,rows)
    cellSize = pixels_per_cell
    blockSize = pixels_per_block    
    blockStride = pixels_per_stride#overlap by one cell
    
    descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,bins)
    return descriptor.compute(image).flatten()
  
def get_image_feature(rimage,imtype):
    if imtype == 'flowers':
        return get_hog_features(rimage)
        
def get_image_features(rimages, imtype):
    if imtype == 'flowers':
        return [get_image_feature(i,imtype) for i in rimages]

    
def get_image_histogram(qimage,mask,bins,regularize=False):
    foremask = mask.astype('uint8')
    fore = (qimage+1)*foremask
    forehist = np.bincount(fore.flatten(),minlength=bins+1)[1:]
    
    backmask = 1-foremask
    back = (qimage+1)*backmask
    backhist = np.bincount(back.flatten(),minlength=bins+1)[1:]
    
    if regularize:
        forehist +=1
        backhist +=1
    
    return forehist,backhist
    
def get_global_histograms(qimages,masks,bins):
    fore_global = np.zeros(bins,'uint64')
    back_global = np.zeros(bins,'uint64')
    for qim,mask in zip(qimages,masks):
        fhist,bhist = get_image_histogram(qim,mask,bins)
        fore_global += fhist.astype('uint64')
        back_global += bhist.astype('uint64')
        
    #don't allow any bin to be zero
    fore_global +=1
    back_global +=1 
    return fore_global,back_global
    
def get_minus_log_prob_pixels(qimage,hist):
    sumhist = hist.sum()
    probs = hist[qimage]/float(sumhist)
    return -np.log(probs)

def get_fidelity_to_histogram(qimage,mask,forehist,backhist):

    #for each pixel in rimage:
    #if foreground get loss to background
    #if background get loss to foreground
    #sumback = backhist.sum()
    #sumfore = forehist.sum()
    rows,cols = qimage.shape
    
    fidmap = np.zeros(qimage.shape) 
    
    #backprobs = backhist[qimage]/float(sumback)
    #backfidelities = -np.log(backprobs)
    backfidelities = get_minus_log_prob_pixels(qimage,backhist)    
    
    #foreprobs = forehist[qimage]/float(sumfore)
    #forefidelities = -np.log(foreprobs)
    forefidelities = get_minus_log_prob_pixels(qimage,forehist)
    
    fidmap[mask] = backfidelities[mask]
    backmask = np.logical_not(mask)
    fidmap[backmask] = forefidelities[backmask]
    
    fidelity = np.sum(fidmap)/(rows*cols)
    return fidelity,fidmap
    

def theta(feat1,feat2,sigma):
    dist = np.linalg.norm(np.array(feat1)-np.array(feat2))
    return np.exp(-dist/ (2*sigma*sigma))
    
def omega1(mask1,mask2):
    total_same =  np.sum(mask1.astype('uint8')==mask2.astype('uint8'))
    npixels = mask1.shape[0]*mask1.shape[1]
    return total_same/float(npixels)
    
def omega2(qim1,qim2,mask1,mask2,bins):
    #get the histograms of mask 2 applied to image 1
    forehist,backhist = get_image_histogram(qim1,mask2,bins,True)
    fidelity,_ = get_fidelity_to_histogram(qim1,mask1,forehist,backhist)
    return fidelity
    
def omega3(qim1,qim2,mask1,mask2,global_forehist,global_backhist):
    im1fidelity,_ = get_fidelity_to_histogram(qim1,mask1,global_forehist,global_backhist)
    im2fidelity,_ = get_fidelity_to_histogram(qim2,mask2,global_forehist,global_backhist)
    return im1fidelity*im2fidelity
    
def K(feat1,feat2,qim1,qim2,mask1,mask2,global_forehist,global_backhist,bins,sigma,beta):
    thetaval = theta(feat1,feat2,sigma)
    o1val = beta[0]*omega1(mask1,mask2)
    o2val = beta[1]*omega2(qim1,qim2,mask1,mask2,bins)
    o3val = beta[2]*omega3(qim1,qim2,mask1,mask2,global_forehist,global_backhist)
    return thetaval*(o1val+o2val+o3val)
    


def get_unary_potentials(testimg,rimages,qimages,imfeatures,masks,global_forehist,\
                            global_backhist,qbins,totalbins,sigma,imtype,\
                            betas,alpha,support_vecs):
    #first resize test image to the correct size and gather features
    rtest = cv2.resize(testimg,qimages[0].shape)
    qtest = get_quantized_image(rtest,qbins)
    feattest = get_image_feature(rtest,imtype)
    #based on test image (j) compared to each support vector image-mask (i)
    
    #the test part of these coefficients, as defined in the paper
    #L(x_{jp} | B_G)
    pf3ip_test = get_minus_log_prob_pixels(qtest,global_backhist)
    #L(X_{jp} | F_G)
    pb3ip_test = get_minus_log_prob_pixels(qtest,global_forehist)
    
    thetas = []
    fore_hists = []
    back_hists = []
    gammas = []
    
    #get infomation for each support vector
    for idx in support_vecs:
        print 'support vec info ',idx
        #same as typical
        thetas.append(theta(feattest,imfeatures[idx],sigma))
        forehist,backhist = get_image_histogram(qtest,masks[idx],totalbins,True)
        fore_hists.append(forehist)
        back_hists.append(backhist)
        
        svecfidelity,_ = get_fidelity_to_histogram(qimages[idx],masks[idx],global_forehist,global_backhist)
        gammas.append(svecfidelity)

    fore_potential = np.zeros(qtest.shape)    
    back_potential = np.zeros(qtest.shape)
    
    for i,idx in enumerate(support_vecs):
        print 'support vec term ',i
        support_theta = thetas[i]
        
        #support_fore = np.zeros(rtest.shape)    
        support_fore = betas[0]*masks[idx]
        support_fore += betas[1] * get_minus_log_prob_pixels(qtest,back_hists[i])
        support_fore += betas[2] * pf3ip_test * gammas[i]
        support_fore *= support_theta
        support_fore *= alpha[idx]
        fore_potential += support_fore
        
        #support_back = np.zeros(rtest.shape)
        support_back = betas[0]*(1-masks[idx])
        support_back += betas[1] * get_minus_log_prob_pixels(qtest,fore_hists[i])
        support_back += betas[2] * pb3ip_test * gammas[i]
        support_back *= support_theta
        support_back *= alpha[idx]
        back_potential += support_back
        
    #now compute foreground potentials
    return fore_potential,back_potential
    
def pixelwise_norms(image):
    return np.sqrt(image[:,:,0]**2 + image[:,:,1]**2 + image[:,:,2]**2)
    
def avg_pixel_difference(rimage):
    right_diffs = rimage[:,1:,:] - rimage[:,:-1,:]
    right_dists = pixelwise_norms(right_diffs)
    
    bottom_diffs = rimage[1:,:,:] - rimage[:-1,:,:]
    bottom_dists = pixelwise_norms(bottom_diffs)
    
    bottomright_diffs = rimage[1:,1:,:] - rimage[:-1,:-1,:]
    bottomright_dists = pixelwise_norms(bottomright_diffs)
    
    all_dists = np.hstack((right_dists.flat,bottom_dists.flat,bottomright_dists.flat))
    return np.mean(all_dists)
    
    
def get_argmax_image(rimage,fore_potential,back_potential,lambda_coef):
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes((rimage.shape[0],rimage.shape[1]))
    #first add the unary potentials    
    graph.add_grid_tedges(nodeids,back_potential,fore_potential)
    
    sigma = avg_pixel_difference(rimage)
    #now add the edgewise smoothing potentials    
    
    #first, right pointing edges
    structure = np.zeros((3,3))
    structure[1,2] = 1
    weights = np.zeros((rimage.shape[0],rimage.shape[1]))
    rightdists = pixelwise_norms(rimage[:,1:,:] - rimage[:,:-1,:])
    weights[:,:-1] = lambda_coef * np.exp(-rightdists/(2*sigma*sigma))
    graph.add_grid_edges(nodeids, structure=structure, weights=weights)
    
    #now, bottom pointing edges
    structure = np.zeros((3,3))
    structure[2,1] = 1
    weights = np.zeros((rimage.shape[0],rimage.shape[1]))
    bottomdists = pixelwise_norms(rimage[1:,:,:] - rimage[:-1,:,:])
    weights[:-1,:] = lambda_coef * np.exp(-bottomdists/(2*sigma*sigma))
    graph.add_grid_edges(nodeids, structure=structure, weights=weights)
    
    #finally, bottom-right pointing edges
    structure = np.zeros((3,3))
    structure[2,2] = 1
    weights = np.zeros((rimage.shape[0],rimage.shape[1]))
    bottomrightdists = pixelwise_norms(rimage[1:,1:,:] - rimage[:-1,:-1,:])
    weights[:-1,:-1] = lambda_coef * (1/np.sqrt(2))*np.exp(-bottomrightdists/(2*sigma*sigma))
    graph.add_grid_edges(nodeids, structure=structure, weights=weights)
    
    #now get the solution!    
    graph.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = graph.get_grid_segments(nodeids)
    return sgm
    
def measure_sa_accuracy(mask,realmask):
    return float(np.sum(mask==realmask))/(mask.shape[0]*mask.shape[1])
    
def measure_so_accuracy(mask,realmask):
    both_obj = np.sum(np.logical_and(mask==1,realmask==1))
    either_obj = np.sum(np.logical_or(mask==1,realmask==1))
    return float(both_obj)/either_obj
    
os.chdir(r"C:\Users\gumpy\Desktop\Class Notes\Advanced Machine Learning\Project\src")


images = []
rimages = []
qimages = []
masks = []
imfeatures = []

#flowers or pedestrians
imtype = 'flowers' 
#use a random order of images or load image 1,2,3... in alphabetical dir order
rand_order = False
n_images = 500
gram = np.zeros((n_images,n_images))
#quantization bins per channel
qbins = 16
totalbins = int(qbins**3)

sigma = .25 #try .1, try .01, try .5, try 1
#start with the Manfredi flowers parameters
betas = (0.2,1.0,0.16)
v = .45
lambda_coef = .24


print 'Loading images and test images'
rimages,masks,_,_ = load_images(imtype,n_images,rand_order)
testimages,testmasks,_,_ = load_images(imtype,200,rand_order,n_images)

print 'Quantizing images'
qimages = get_quantized_images(rimages,qbins)
print 'Extracting image features'
imfeatures = get_image_features(rimages,imtype)
print 'Getting global color histogram'
fore_global_hist, back_global_hist = get_global_histograms(qimages,masks,totalbins)
print 'Done'

gram1 = np.load('200Gram.npy')

gram = np.zeros((n_images,n_images))

print 'Computing gram matrix'
for i in range(n_images):
    print "Row ",i
    for j in range(n_images):
        if i <200 and j<200:
            gram[i,j] = gram1[i,j]
            continue
        
        feat1,feat2 = imfeatures[i],imfeatures[j]
        qim1,qim2 = qimages[i],qimages[j]
        mask1,mask2 = masks[i],masks[j]
        val = K(feat1,feat2,qim1,qim2,mask1,mask2,\
                    fore_global_hist,back_global_hist,totalbins,sigma,betas)
        gram[i,j] = val



ocSVM = svm.OneClassSVM(kernel='precomputed',nu=v)
ocSVM.fit(gram)
alpha = ocSVM.dual_coef_.flatten()
support_vecs = ocSVM.support_
print len(support_vecs)


#Test unary potentials
total_ims = 0
total_a_acc = 0.0
total_o_acc = 0.0

for i in range(len(testimages)):
    fore,back = get_unary_potentials(testimages[i],rimages,qimages,imfeatures,masks,fore_global_hist,\
                                back_global_hist,qbins,totalbins,sigma,imtype,\
                                betas,alpha,support_vecs)
                                
    rtest = cv2.resize(testimages[i],qimages[0].shape)
    rtestmask = testmasks[i]
    
    amax = get_argmax_image(rtest,fore,back,lambda_coef)

    rmasked = rtest.copy()
    rmasked[amax==0]/=5
    
    a_acc = measure_sa_accuracy(amax,rtestmask)
    o_acc = measure_so_accuracy(amax,rtestmask)
    
    print "s_a Image accuracy is ",a_acc
    total_a_acc += a_acc
    total_o_acc += o_acc
    total_ims+=1
    print "Average s_a accuracy is ",total_a_acc/total_ims
    print "Average s_o accuracy is ",total_o_acc/total_ims
    
    f1 = plt.figure()
    plt.imshow(fore-back)
    plt.colorbar()
    
    cv2.imshow('Original', rtest)
    cv2.imshow('Masked image',rmasked)
    cv2.waitKey()
    plt.close(f1)

    #plt.close(f2)
    



"""
#test histogram fidelities
maxfidelity = 0
minfidelity = 9999999999999999999999999
for i in range(n_images):
    start = time.clock()
    fidelity,fidmap = get_fidelity_to_histogram(qimages[i],masks[i],fore_global_hist,back_global_hist)
    print "took ", time.clock()-start," seconds"    
    print fidelity
    
    if fidelity > maxfidelity:
        print "max fidelity is  now", fidelity
        maxfidelity = fidelity
        cv2.imshow("High Fidelity",rimages[i])
        
        plt.close()
        plt.imshow(fidmap)
        plt.colorbar()
        
        cv2.waitKey()
    if fidelity < minfidelity:
        print "min fidelity is  now", fidelity
        minfidelity = fidelity
        cv2.imshow("Low Fidelity",rimages[i])
        
        plt.close()
        plt.imshow(fidmap)
        plt.colorbar()
        cv2.waitKey()
"""


#TEST OF THETA KERNEL
"""
for i in range(n_images):
    max_sim = 0
    best_idx = -1
    for j in range(i+1,n_images):
        sim = theta(imfeatures[i],imfeatures[j],sigma)
        if sim > max_sim:
            max_sim = sim
            best_idx = j
    cv2.imshow("image 1",rimages[i])
    cv2.imshow("image 2",rimages[best_idx])
    c = cv2.waitKey()
"""

#TEST omega1 kernel
"""
for i in range(n_images):
    max_sim = 0
    best_idx = -1
    for j in range(i+1,n_images):
        sim = omega1(masks[i],masks[j])
        if sim > max_sim:
            max_sim = sim
            best_idx = j
    cv2.imshow("image 1",rimages[i])
    cv2.imshow("image 2",rimages[best_idx])
    
    c = cv2.waitKey()
    print max_sim
"""

"""
#TEST omega2 kernel
for i in range(n_images):
    max_sim = 0
    best_idx = -1
    bestmap = None
    
    for j in range(i+1,n_images):
        #start = time.clock()
        sim = omega2(qimages[i],qimages[j],masks[i],masks[j],totalbins)
        print sim
        #print time.clock()-start," seconds to compute omega2"
        if sim > max_sim:
            max_sim = sim
            best_idx = j
    print "Best value is ",max_sim
    cv2.imshow("image 1",rimages[i])
    cv2.imshow("image 2",rimages[best_idx])
    
    c = cv2.waitKey()
    print max_sim
"""

"""
#fore_global_hist
#back_global_hist
#totalbins = int(qbins**3)
#def K(feat1,feat2,qim1,qim2,mask1,mask2,global_forehist,global_backhist,bins,sigma,beta):
#TEST K
for i in range(n_images):
    max_sim = 0
    best_idx = -1
    bestmap = None
    
    for j in range(i+1,n_images):
        start = time.clock()
        feat1,feat2 = imfeatures[i],imfeatures[j]
        qim1,qim2 = qimages[i],qimages[j]
        mask1,mask2 = masks[i],masks[j]
        sim = K(feat1,feat2,qim1,qim2,mask1,mask2,fore_global_hist,back_global_hist,totalbins,sigma,beta)
        print sim
        print time.clock()-start," seconds to compute K"
        
        if sim > max_sim:
            max_sim = sim
            best_idx = j
    print "Best value is ",max_sim
    cv2.imshow("image 1",rimages[i])
    cv2.imshow("image 2",rimages[best_idx])
    
    c = cv2.waitKey()
    print max_sim
"""