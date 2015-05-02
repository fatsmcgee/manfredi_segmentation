import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def mask_from_image(image,imtype):
    if imtype == 'flowers':
        return np.logical_and(image[:,:,2]==128,image[:,:,1]==0)

#1. rimages,masks = load_images('flowers',100)
def load_images(imtype, n_images,rand_order):
    
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
        
    for im_f,m_f in i_m_fs[:n_images]:
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
    
def get_image_features(rimages, imtype):
    if imtype == 'flowers':
        return [get_hog_features(i) for i in rimages]

    
def get_image_histogram(qimage,mask,bins):
    foremask = mask.astype('uint8')
    fore = (qimage+1)*foremask
    forehist = np.bincount(fore.flatten(),minlength=bins+1)[1:]
    
    backmask = 1-foremask
    back = (qimage+1)*backmask
    backhist = np.bincount(back.flatten(),minlength=bins+1)[1:]
    
    return forehist,backhist
    
def get_global_histograms(qimages,masks,bins):
    fore_global = np.zeros(bins,'uint64')
    back_global = np.zeros(bins,'uint64')
    for qim,mask in zip(qimages,masks):
        fhist,bhist = get_image_histogram(qim,mask,bins)
        fore_global += fhist
        back_global += bhist
        
    #don't allow any bin to be zero
    fore_global +=1
    back_global +=1 
    return fore_global,back_global
    

def get_fidelity_to_histogram(qimage,mask,forehist,backhist):

    #for each pixel in rimage:
    #if foreground get loss to background
    #if background get loss to foreground
    sumback = backhist.sum()
    sumfore = forehist.sum()
    rows,cols = qimage.shape
    
    fidmap = np.zeros(qimage.shape) 
    
    backprobs = backhist[qimage]/float(sumback)
    backfidelities = -np.log(backprobs)
    
    foreprobs = forehist[qimage]/float(sumfore)
    forefidelities = -np.log(foreprobs)
    
    fidmap[mask] = backfidelities[mask]
    backmask = np.logical_not(mask)
    fidmap[backmask] = forefidelities[backmask]
    
    return np.sum(fidmap)/(rows*cols),fidmap
    

def theta(feat1,feat2,sigma):
    dist = np.linalg.norm(np.array(feat1)-np.array(feat2))
    return np.exp(-dist/ (2*sigma*sigma))
    
def omega1(mask1,mask2):
    total_same =  np.sum(mask1.astype('uint8')==mask2.astype('uint8'))
    npixels = mask1.shape[0]*mask1.shape[1]
    return total_same/float(npixels)
    
os.chdir(r"C:\Users\gumpy\Desktop\Class Notes\Advanced Machine Learning\Project\src")


images = []
rimages = []
qimages = []
masks = []
imfeatures = []

#flowers or pedestrians
imtype = 'flowers' 
#use a random order of images or load image 1,2,3... in alphabetical dir order
rand_order = True
n_images = 300
#quantization bins per channel
qbins = 16
total_color_bins = int(qbins**3)

print 'Loading images'
rimages,masks,imfiles,mfiles = load_images(imtype,n_images,rand_order)
print 'Quantizing images'
qimages = get_quantized_images(rimages,qbins)
print 'Extracting image features'
imfeatures = get_image_features(rimages,imtype)
print 'Getting global color histogram'
fore_global_hist, back_global_hist = get_global_histograms(qimages,masks,total_color_bins)
print 'Done'

sigma = .25 #try .1, try .01, try .5, try 1

#test histogram fidelities

maxfidelity = 0
minfidelity = 9999999999999999999999999
for i in range(n_images):
    fidelity,fidmap = get_fidelity_to_histogram(qimages[i],masks[i],fore_global_hist,back_global_hist)
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