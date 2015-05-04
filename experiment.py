import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import multiprocessing
from sklearn import svm
import maxflow

def mask_from_image(image,imtype):
    if imtype == 'flowers':
        return np.logical_and(image[:,:,2]==128,image[:,:,1]==0)

def get_image_paths(imtype,randorder=False):  
    i_m_fs = []
    
    if imtype == 'flowers':
        im_fs = os.listdir("../flower_images")
        m_fs = ['../flower_segments/' + f[:-4] + ".png"\
                            for f in im_fs]
        im_fs = ['../flower_images/' + f for f in im_fs]
        i_m_fs = [t for t in zip(im_fs,m_fs) if os.path.isfile(t[1])]
        
    if randorder:
        np.random.shuffle(i_m_fs)
        
    return [t[0] for t in i_m_fs], [t[1] for t in i_m_fs]
    
#1. rimages,masks = load_images('flowers',100)
def load_images(imtype,n,impaths,maskpaths):
    
    rimages = []
    masks = []
    labeled = []
    newsize = None
    
    if imtype == 'flowers':
        newsize = (256,256)
        
    for im_f,m_f in zip(impaths,maskpaths)[:n]:
        im = cv2.imread(im_f)
        rimages.append(cv2.resize(im,newsize))
        
        m = cv2.imread(m_f)
        m = cv2.resize(m,newsize)
        mask = mask_from_image(m,imtype)
        masks.append(mask)
        
        if imtype == 'flowers': #in flowers data set, not all pixels are labeled
            label = np.logical_and(m[:,:,0]==0,m[:,:,1]==0)
            label = np.logical_and(label,m[:,:,2]==0)
            label = np.logical_not(label)
            labeled.append(label)
        else:
            labeled.append(np.ones(mask.shape).astype('bool'))
        
                
    return rimages,masks,labeled
        
#take each channel in the image, and put it in a bin uniformly between 0 and qbins
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
    
#as specified in paper:
#
def get_flower_hog_features(image):
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
        return get_flower_hog_features(rimage)
        
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

#Calculate \sum_{p=1}^P L(x_{ip} | y_{ip},F,B)
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

def get_kernels(feat1,feat2,qim1,qim2,mask1,mask2,global_forehist,global_backhist,bins,sigma):
    thetaval = theta(feat1,feat2,sigma)
    o1val = omega1(mask1,mask2)
    o2val = omega2(qim1,qim2,mask1,mask2,bins)
    o3val = omega3(qim1,qim2,mask1,mask2,global_forehist,global_backhist)
    #print 'values be ',thetaval,o1val,o2val,o3val
    return thetaval,o1val,o2val,o3val
    
    
def K(feat1,feat2,qim1,qim2,mask1,mask2,global_forehist,global_backhist,bins,sigma,beta):
    thetaval = theta(feat1,feat2,sigma)
    o1val = beta[0]*omega1(mask1,mask2)
    o2val = beta[1]*omega2(qim1,qim2,mask1,mask2,bins)
    o3val = beta[2]*omega3(qim1,qim2,mask1,mask2,global_forehist,global_backhist)
    return thetaval*(o1val+o2val+o3val)
   
#calculate part of the gram matrix and save to disk
#done this way to allow for multiprocessing
def get_partial_kernels(n_images,rowstart,rowend,imfeatures,qimages,masks\
                    ,fore_global_hist,back_global_hist,totalbins,sigma,betas):
                        
        kernels = np.zeros((rowend-rowstart+1,n_images,4))
        for i in range(rowstart,rowend+1):
            sys.stdout.write( "Row {0} out of {1}\n".format(i,rowend))
            sys.stdout.flush()
            for j in range(n_images):
                feat1,feat2 = imfeatures[i],imfeatures[j]
                qim1,qim2 = qimages[i],qimages[j]
                mask1,mask2 = masks[i],masks[j]
                theta,omega1,omega2,omega3 = get_kernels(feat1,feat2,qim1,qim2,mask1,mask2,\
                            fore_global_hist,back_global_hist,totalbins,sigma)
                kernels[i-rowstart,j,0] = theta
                kernels[i-rowstart,j,1] = omega1
                kernels[i-rowstart,j,2] = omega2
                kernels[i-rowstart,j,3] = omega3
        np.save('subkernels{0}.npy'.format(rowstart),kernels)

def get_all_kernels(n_processes,n_images,imfeatures,qimages,masks\
                    ,fore_global_hist,back_global_hist,totalbins,sigma,betas):

    jobs = []
    rowstarts = []
    chunk_size = int(n_images/n_processes)
    for rs in range(0,n_images,chunk_size):
        rowstarts.append(rs)
        rowend = min(rs+chunk_size-1,n_images-1)
            
        args = (n_images,rs,rowend,imfeatures,qimages,masks,\
                fore_global_hist,back_global_hist,totalbins,sigma,betas)
        proc = multiprocessing.Process(target=get_partial_kernels, args=args)
        jobs.append(proc)
        proc.start()
        
    for proc in jobs:
        proc.join()
        #now join the resulting matrices
    part_grams = tuple(np.load('subkernels{0}.npy'.format(rs)) for rs in rowstarts)
    kernels = np.vstack(part_grams)
    
    np.save('kernels.npy',kernels)
    return kernels
    
#used for crossvalidating over simga
#theta is very inexpensive to replace    
def replace_theta(kernels,newsigma):
    n_images = kernels.shape[0]
    newkernels = kernels.copy()
    
    for i in range(n_images):
        for j in range(n_images):
            newkernels[:,:,0] = theta(imfeatures[i],imfeatures[j],newsigma)
            
    return newkernels
    
def get_graham_matrix(kernels,betas):
    theta = kernels[:,:,0]
    omega1,omega2,omega3 = kernels[:,:,1],kernels[:,:,2],kernels[:,:,3]
    return theta * (betas[0]*omega1 + betas[1]*omega2 + betas[2]*omega3)

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
        support_fore *= alpha[i]
        fore_potential += support_fore
        
        #support_back = np.zeros(rtest.shape)
        support_back = betas[0]*(1-masks[idx])
        support_back += betas[1] * get_minus_log_prob_pixels(qtest,fore_hists[i])
        support_back += betas[2] * pb3ip_test * gammas[i]
        support_back *= support_theta
        support_back *= alpha[i]
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
    
def measure_sa_accuracy(mask,realmask,labeled):
    same_in_label = np.logical_and(labeled, mask==realmask)
    return float(np.sum(same_in_label))/np.sum(labeled)
    
def measure_so_accuracy(mask,realmask,labeled):
    both_obj = np.logical_and(labeled,np.logical_and(mask==1,realmask==1))
    either_obj = np.logical_and(labeled, np.logical_or(mask==1,realmask==1))
    return float(np.sum(both_obj))/np.sum(either_obj)

if __name__ == '__main__':    
    
    #flowers or pedestrians
    imtype = 'flowers' 
    #use a random order of images or load image 1,2,3... in alphabetical dir order
    #choose an arbitrary seed so we get the same behavior each time
    np.random.seed(10)
    rand_order = True
    #cfc
    interactive = True
    n_images = 600
    n_testimages = 229
    gram = np.zeros((n_images,n_images))
    #quantization bins per channel
    qbins = 16
    totalbins = int(qbins**3)
    
    sigma = .25 #try .1, try .01, try .5, try 1
    #start with the Manfredi flowers parameters
    betas = (0.2,1.0,0.16)
    v = .45
    lambda_coef = .24
    #cfc
    n_processes = 4
    
    
    print 'Loading images and test images'
    impaths,maskpaths = get_image_paths(imtype,rand_order)
    allimages,allmasks,all_labels = load_images(imtype,n_images+n_testimages,impaths,maskpaths)
    
    rimages,masks = allimages[:n_images], allmasks[:n_images]
    testimages,testmasks,testlabels = allimages[n_images:],allmasks[n_images:],all_labels[n_images:]
    
    print 'Quantizing images'
    qimages = get_quantized_images(rimages,qbins)
    print 'Extracting image features'
    imfeatures = get_image_features(rimages,imtype)
    print 'Getting global color histogram'
    fore_global_hist, back_global_hist = get_global_histograms(qimages,masks,totalbins)
    print 'Done'
    

    
    #kernels = get_all_kernels(n_processes,n_images,imfeatures,qimages,masks\
     #               ,fore_global_hist,back_global_hist,totalbins,sigma,betas)
    kernels = np.load('600KERNEL.npy')
    
    gram = get_graham_matrix(kernels,betas)
    #gram = np.load('600GRAMNORAND.npy')
     
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
        rtestlabeled = testlabels[i]
        
        amax = get_argmax_image(rtest,fore,back,lambda_coef)
    
        rmasked = rtest.copy()
        rmasked[amax==0]/=10
        
        rgroundtruth = rtest.copy()
        rgroundtruth[rtestmask==0]/=10
        
        a_acc = measure_sa_accuracy(amax,rtestmask,rtestlabeled)
        o_acc = measure_so_accuracy(amax,rtestmask,rtestlabeled)
        
        print "s_a Image accuracy is ",a_acc
        print "s_o Image accuracy is ",o_acc
        total_a_acc += a_acc
        total_o_acc += o_acc
        total_ims+=1
        print "Average s_a accuracy is ",total_a_acc/total_ims
        print "Average s_o accuracy is ",total_o_acc/total_ims
        
        if interactive:
            f1 = plt.figure()
            plt.imshow(fore-back)
            plt.colorbar()
    
            cv2.imshow('Original', rtest)
            cv2.imshow('Argmax Masked image',rmasked)
            cv2.imshow('Ground Truth Masked Image',rgroundtruth)
            
            cv2.waitKey()
            plt.close(f1)
    
        #plt.close(f2)
    
    
    
