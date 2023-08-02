import torch
import matplotlib.pyplot as plt

import os
import torch.nn.functional as F
from torchvision import transforms
import mxnet as mx
import numpy as np
import numbers

from PIL import Image
from tqdm import tqdm

import net
import pickle
import gc


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#tensor_path = './experiments/ir101_ms1mv2_adaface_01-28_0/epoch=29-step=341189.ckpt' ## ada
tensor_path = './experiments/ir101_ada_limit_06-12_2/epoch=34-step=398054.ckpt' ## adalimit


def main():
    checkpoint  = torch.load(tensor_path)

    def l2_norm(input,axis=1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input, norm)
        return output

    root_dir = '/home/taes/face_data/ms1m/'
    path_imgrec = os.path.join(root_dir, 'train.rec')
    path_imgidx = os.path.join(root_dir, 'train.idx')
    path_imglst = os.path.join(root_dir, 'train.lst')
    record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    imgidx = np.array(list(record.keys))

    model = net.build_model('ir_101')
    model.load_state_dict({key.replace('model.', ''):val for key,val in checkpoint['state_dict'].items() if 'model.' in key})
    train_transform = transforms.Compose([
    #          transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    model.eval()
    model.cuda()

    #try:
    #    labels = []
    #    for idx in tqdm(imgidx):
    #        s = record.read_idx(idx)
    #        header, img = mx.recordio.unpack(s)
    #        label = header.label
    #        if not isinstance(label, numbers.Number): 
    #            label = label[0]
    #        labels.append(label)
    #        img = mx.image.imdecode(img).asnumpy() ## TODO remove 
    #except Exception as e:
    #    print("err1")
    #    pass
    s = record.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        labels = np.array(range(1,int(header.label[0])))
    else:
        labels = np.array(list(record.keys))

    print(len(labels))


    label_num = {}
    img_idx = {}
    for i,idx in zip(labels,imgidx):
        if i<70722:
            
            if i in label_num.keys():
                label_num[i] = label_num[i] + 1
                img_idx[i].append(idx)
            else:
                label_num[i] = 1
                img_idx[i] = [idx]

    s = [int(i[1]) for i in label_num.items()]
    sr = [int(k) for k, v in sorted(label_num.items(), key=lambda item: item[1])]

    kernel_norm = l2_norm(checkpoint['state_dict']['head.kernel'],axis=0).cuda()
    
    batch_size = 256 
    ms1m_all = {}
    flag = 1
    for idx in tqdm(range(int(len(imgidx)/batch_size))):
        
        img_batch = []
        label_batch = []
        for batch in range(batch_size):
            try:
                
                s = record.read_idx(imgidx[idx*batch_size+batch])
                header, img = mx.recordio.unpack(s)
                label = header.label
                if not isinstance(label, numbers.Number): 
                    label = label[0]
                
                img = mx.image.imdecode(img).asnumpy()
                img = Image.fromarray(np.asarray(img)[:, :, ::-1])
                
                
                
                label_batch.append(int(label))
                img_batch.append(train_transform(img).unsqueeze(0))
            
            except Exception as e:
                print(e)
                flag = -1
                break
        
        if flag==-1:
            break
        
        img_tensors = torch.cat(img_batch,0)        
        # if label < 10572:

        feat,norm = model(img_tensors.cuda())
        predict = torch.mm(feat,kernel_norm)
        predict = np.array(predict.detach().cpu())
        norm = np.array(norm.detach().cpu())        
        

        for label_idx in range(len(label_batch)):
            
            label_ = label_batch[label_idx]
            cos_value = float(predict[label_idx][label_])                
            norm_value = norm[label_idx][0]
            max_pred_idx = np.max(predict[label_idx])
            max_pred_val = np.argmax(predict[label_idx])
            
            stack_res = [cos_value,norm_value,max_pred_idx,max_pred_val]

            if label_ in ms1m_all.keys():
                ms1m_all[label_].append(stack_res)
            else:
                ms1m_all[label_] = [stack_res]

        predict = None
        norm = None
        feat = None
        gc.collect()
        torch.cuda.empty_cache()
                    




    with open('for_gst_adalimit25_ms1m.pickle', 'wb') as f:
        pickle.dump(ms1m_all, f)



if __name__ == '__main__':
    main()
