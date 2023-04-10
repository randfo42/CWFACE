import torch
import os
import torch.nn.functional as F
import mxnet as mx
import numpy as np
import pandas as pd
import numbers

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pickle

import net

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#dict_name = 'samples_feature_norm.pickle'
#dict_name = 'cosine_distance.pickle'
#dict_name = 'samples_feature_norm_ir50_cw.pickle'
dict_name = 'samples_feature_norm_ir50_cwl.pickle'

#tensor_path = './experiments/ir50_casia_adaface_01-02_3/last.ckpt'
#tensor_path = './experiments/ir50_casia_wc_weight_01-20_________18/last.ckpt'
tensor_path = './experiments/ir50_casia_wc_weight_test_01-28_9/last.ckpt'
root_dir = './faces_webface_112x112/'


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input,norm)
    return output

if __name__ == '__main__':
    
    checkpoint  = torch.load(tensor_path)
    path_imgrec = os.path.join(root_dir, 'train.rec')
    path_imgidx = os.path.join(root_dir, 'train.idx')
    path_imglst = os.path.join(root_dir, 'train.lst')
    
    record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    
    model = net.build_model()
    model.load_state_dict({key.replace('model.', ''):val for key,val in checkpoint['state_dict'].items() if 'model.' in key})
    
    train_transform = transforms.Compose([
#          transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ])
    
    model.eval()
    model.cuda()
    
    record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    imgidx = np.array(list(record.keys))
    
    samples_feature_norm = {}

    
    kernel_norm = l2_norm(checkpoint['state_dict']['head.kernel'],axis=0).cuda()

    for idx in tqdm(imgidx):
        s = record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number): 
            label = label[0]
        if label < 10572:
            try:
                sample = mx.image.imdecode(img).asnumpy()
                sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
            except:
                continue
            
            ## feature norm
            if False:
                res = float(model(train_transform(sample).unsqueeze(0))[1][0][0].cpu())
    #         print(samples_feature_norm[int(label)])
                if int(label) in samples_feature_norm.keys():
                    samples_feature_norm[int(label)].append(res)
                else:
                    samples_feature_norm[int(label)] = [res]
            
            # cosine dist
            else:
                
                feat,norm = model(train_transform(sample).unsqueeze(0).cuda())
                res = torch.mm(feat,kernel_norm)[0,int(label)]
                res = float((res.cpu()))
                #print(res)
                
                if int(label) in samples_feature_norm.keys():
                    samples_feature_norm[int(label)].append(res)
                else:
                    samples_feature_norm[int(label)] = [res]



    with open(dict_name,'wb') as fw:
        pickle.dump(samples_feature_norm, fw)


    
    
    
    
