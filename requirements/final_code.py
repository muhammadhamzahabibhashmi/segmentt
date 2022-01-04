import os
import warnings
warnings.filterwarnings("ignore")
import json
import csv
import numpy as np
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms as trn

from mit_semseg.utils import find_recursive
from mit_semseg.lib.utils import as_numpy
from mit_semseg.config import cfg

from sphericalpadding import *
from segm.model.utils import inference
from segm.model.factory import load_model
from pathlib import Path
import segm.utils.torch as ptu
from segm.data.utils import STATS
import torchvision.transforms.functional as F

import time
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb
from scipy.ndimage import map_coordinates

colors = loadmat('data/color150.mat')['colors']
color_codes_dict={}
iter=1

for x in colors:
    color_codes_dict[tuple(int(y) for y in x)]=iter
    iter+=1

def pyfunc(x):
    return color_codes_dict[tuple(x)]

vfunc = np.vectorize(pyfunc, signature='(3)->()')
 
def plot_contours(new_maskss , image_names):

    images_names=image_names
    ratio=1
    n_classes=150
    smooth_alpha = 0.0012
    polygons=[]
    class_uv=[]
    back_g_faces=[]

    for index,im_no in enumerate([3,2,4,0,1,5]):
        img=Image.fromarray(new_maskss[im_no].astype(np.uint8))

        img = np.array(img)
        # translate 3d color vectors to scalar ints [0, ..., n_classes]
        mask = vfunc(img)
        h, w = mask.shape
        back_g=np.zeros((h,w,3))

        mask_classes=list(np.unique(mask))
        for class_k in mask_classes:
            class_k=int(class_k)

            contours=[]
            approxes=[]
            new_mask = np.zeros_like(mask, dtype='uint8')
            # get binary mask for current class
            new_mask[mask == class_k] = 1
            new_mask[mask != class_k] = 0
            uvgrid = equirect_uvgrid(h, w)
            contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                epsilon = smooth_alpha * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                back_g=cv2.drawContours(back_g, cnt, -1, (255,255,255), 5)

                uv_approx = []
                for i in range(len(approx)):
                    y, x = approx[i][0]
                    u=uvgrid[x, y][0]
                    v=uvgrid[x, y][1]

                    if index>0 and index !=4 and index!=5:
                        uv_approx.append([(u+int(index*np.pi)).tolist(),v.tolist()])

                    elif index == 4:
                        uv_approx.append([(u+int(np.pi)).tolist(),v.tolist()])

                    elif index == 5:
                        uv_approx.append([(u+int(np.pi)).tolist(),v.tolist()])

                    else:
                        uv_approx.append([u.tolist(),v.tolist()])

                polygons.append(uv_approx)
                class_uv.append(class_k)
        back_g_faces.append(back_g)

    combined_imgae=np.hstack((back_g_faces[0],back_g_faces[1]))
    plt.imshow(combined_imgae)
    plt.show()
    class_polygons_dict = {}
    for x, y in zip(class_uv, polygons):
        class_polygons_dict.setdefault(x, []).append(y)

######################################################## combine all the u,v points ####################################################

    json_dict = dict.fromkeys(['Image_name','Classes'])
    json_dict['Image_name']=images_names
    json_dict['Classes']=class_polygons_dict

    with open('uv_polygons_face1_2offsetpi.json', 'w') as fp:
        json.dump(json_dict, fp)  
    
    return fp
  
names = {}

with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def equirect_facetype(h, w):
    
    '''
    0F 1R 2B 3L 4U 5D
    '''

    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def visualize_result_optimal(data, pred, cfg, results):
    (img, info) = data
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            pass
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    results['rimgs'].append(pred_color)

def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)

def compare_roll_ori(result,result_roll):
    updated=[]
    image_names=result_roll['image_name'][0]
    for i in range(6):

        roll_img=result_roll['rimgs'][i]
        _img=result['rimgs'][i]

        a=result['probs'][i].cpu().numpy()
        b=result_roll['probs'][i].cpu().numpy()
        mask1=np.array(b>a).reshape(840,840).astype(np.uint8)

        img2 = np.zeros_like(roll_img)
        img2[:,:,0] = mask1
        img2[:,:,1] = mask1
        img2[:,:,2] = mask1
        

        mask2=np.array(a>=b).reshape(840,840).astype(np.uint8)
        img3 = np.zeros_like(_img)

        img3[:,:,0] = mask2
        img3[:,:,1] = mask2
        img3[:,:,2] = mask2

        new_replace=img3*_img + img2*roll_img
        updated.append(new_replace)
    return updated,image_names

def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2
    return np.stack(np.meshgrid(u, v), axis=-1)


class Sphegmenter():
    #@markdown #the **Sphegmenter** class has 5 function:
    #@markdown **1)** the **initialization** function declares the initial variables:
    def __init__(self, model_path='seg_tiny_mask/checkpoint.pth'):
      
        self.batch_size = 6
        self.CUDA=True
        padding = 100
        self.cube_pad = SpherePad(padding)
        # Network Builders
        ptu.set_gpu_mode(self.CUDA)

        model_dir = Path(model_path).parent
        self.model, self.variant = load_model(model_path)
        self.model.to(ptu.device)
        normalization_name = self.variant["dataset_kwargs"]["normalization"]
        self.normalization = STATS[normalization_name]
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #@markdown **2)** the **predict** function will be used to make inference:
    def predict(self,image_path):
        Im = Image.open(image_path).convert('RGB')
        cube_size_ori = Im.width//4
        output_w_ori = Im.width
        output_h_ori = Im.height
        cube_fov=90
        c2e = Cube2Equirec(self.batch_size, cube_size_ori, output_h_ori, output_w_ori, cube_fov, self.CUDA)
        e2c = Equirec2Cube(self.batch_size, output_h_ori, output_w_ori, cube_size_ori, cube_fov, self.CUDA)

        self.merge = np.zeros((Im.height, Im.width, 3))
        results=self.segment(Im,image_path,e2c,cube_size_ori)

        torch.cuda.empty_cache() 
        im = np.roll(np.array(Im), Im.width//4, axis=1)
        im=Image.fromarray(im)
        cube_size_roll = Im.width//4
        output_w_roll = Im.width
        output_h_roll = Im.height      
        result_roll = self.segment(im,image_path,e2c,cube_size_roll)

        starttime_=time.time()
        new_maskss,images_names=compare_roll_ori(results,result_roll)
        print('Comparing rolled and original time',time.time()-starttime_)
        plot_contours(new_maskss,images_names)
        print('plotting contours X,y to u,v conversion time',time.time()-starttime_)
        return
    #@markdown **3)** the **segment()** function performs segmentation:
    def segment(self,Im,image_path,e2c,cube_size_roll):
        ow, oh = Im.size
        im = np.array(Im)
        
        if self.CUDA:
            im_tensor = Variable(torch.cuda.FloatTensor(im).unsqueeze(0).transpose(1, 3).transpose(2, 3))
        else:
            im_tensor = Variable(torch.FloatTensor(im).unsqueeze(0).transpose(1, 3).transpose(2, 3))
        im_cube = e2c.ToCubeTensor(im_tensor)
        padded_original = self.cube_pad(torch.stack([x.squeeze(0) for x in im_cube]))
        padded_faces = [padded_original[i].permute(1,2,0).cpu().numpy().astype(np.uint8) for i in range(6)]

        del padded_original
        del im_cube
        del im_tensor
        image_name=image_path.split('/')[-1].split('.')[0]
        pf_folder_name="padded_faces"+image_name
        os.makedirs(pf_folder_name, exist_ok=True)  # succeeds even if directory exists.
        for i, side in enumerate(['back', 'bottom', 'front', 'left', 'right', 'top']):

            Image.fromarray(padded_faces[i]).save(pf_folder_name+'/'+side+'.jpg')
        if os.path.isdir(pf_folder_name):
            imgs = find_recursive(pf_folder_name)
        else:
            imgs = [pf_folder_name]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."
        list_test = [{'fpath_img': x} for x in imgs]

        results = {"probs":[],
                   "preds":[],
                   "rimgs":[],
                   "image_name":[]}
        
        for img_path in imgs:
          with torch.no_grad():
                pil_im = Image.open(img_path).convert('RGB')
                im_meta = dict(flip=False)
                im = F.pil_to_tensor(pil_im).float() / 255
                im = F.normalize(im, self.normalization["mean"], self.normalization["std"])
                segSize=(im.shape[1],
                          im.shape[2])
                im = im.pin_memory().to(ptu.device).unsqueeze(0)
                pred_tmp = inference(self.model,[im],[im_meta],ori_shape=im.shape[2:4],window_size=self.variant["inference_kwargs"]["window_size"],window_stride=self.variant["inference_kwargs"]["window_stride"],batch_size=2)
                
                cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)

                seg_map = pred_tmp.argmax(0, keepdim=True)
                seg_rgb = seg_to_rgb(seg_map, cat_colors)
                seg_rgb = (255 * seg_rgb.squeeze().cpu().numpy()).astype(np.uint8)

                pred_tmp=pred_tmp.unsqueeze(0)

                prob, pred = torch.max(pred_tmp[0].unsqueeze(0), dim=1)

                pred=pred.squeeze(0)
                visualize_result_optimal( (np.transpose(np.array(pil_im), (1, 0, 2)), img_path),as_numpy(pred.cpu()),cfg,results)

                results['preds'].append( pred)
                results['probs'].append( prob.squeeze(0))
                results["image_name"].append(image_name)


        seg_folder_name="seg_results"+image_path.split('/')[-1].split('.')[0]
        os.makedirs(seg_folder_name, exist_ok=True) 
        for i, side in enumerate(["back", "bottom", "front", "left", "right", "top"]): #[3,2,4,0]
            Image.fromarray((results['rimgs'][i][100:-100, 100:-100]*255).astype(np.uint8)).save(seg_folder_name+'/'+side+'.jpg')
        return results
    #@markdown **4)** the **visialize()** function is used to visualise the results:
    def visualize(self, pred):
        pred = np.int32(pred)
        pred_color = colorEncode(pred, colors).astype(np.uint8)
        return pred_color

    #@markdown **5)** the **combined_results()** function is used to comine the results:
    def combined_results(self, input_path, result, cube_size,output_w,output_h,c2e,padding=100, roll=False):
        batch_size = 6
        cube_fov = 90
        CUDA = True

        seg_cube = torch.stack([torch.from_numpy(np.array(result['rimgs'][i])[padding:-padding, padding:-padding]).permute(2, 0, 1).float().cuda() for i in range(6)])
        im_equi_ori = c2e.ToEquirecTensor(seg_cube)
        im_equi_ori = im_equi_ori.to(torch.uint8).transpose(1, 3).transpose(1, 2).cpu().numpy()[0]#.astype(np.uint8)

        seg_probs = torch.stack([torch.cat(
                        [result['probs'][i][padding:-padding, padding:-padding].unsqueeze(2), 
                        result['probs'][i][padding:-padding, padding:-padding].unsqueeze(2), 
                        result['probs'][i][padding:-padding, padding:-padding].unsqueeze(2)],dim=2).permute(2, 0, 1)
                for i in range(6)
            ]
        )
        prob_equi_ori = c2e.ToEquirecTensor(seg_probs)
        prob_equi_ori = prob_equi_ori.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(np.uint8)

        if roll:
            im_equi_ori = np.roll(im_equi_ori, -output_w//4, axis=1)
            prob_equi_ori = np.roll(prob_equi_ori, -output_w//4, axis=1)

        return im_equi_ori,prob_equi_ori,None


# starttime=time.time()


model_location= 'seg_tiny_mask/checkpoint.pth'

#@param {type:"string"}

s = Sphegmenter(model_location)


# print('Model loading time',time.time()-starttime)


# starttime=time.time()


image_location= 'images/images_camera-h-02.jpg' 

#@param {type:"string"}

result = s.predict(image_location)

# print('Total Inference time (seconds)',time.time()-starttime)