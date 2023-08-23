from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn
import numpy as np

def build_head(head_type,
               embedding_size,
               class_num,
               m,
               t_alpha,
               h,
               s,
            
               ):
    

    print(head_type)
    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )
    elif head_type == 'cwface':
        head = CWFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )
    elif head_type == 'cwlface':
        head = CWLFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'cwcface':
        head = CWCFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'lamaface':
        head = LAMAFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'cwlnface':
        head = CWLNFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'adawindexface':
        head = AdaWIndexFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'adawindexface':
        head = AdaWIndexFace(embedding_size=embedding_size,
               classnum=class_num,
               m=m,
               h=h,
               s=s,
               t_alpha=t_alpha,
               )
    elif head_type == 'adasface':
        head = AdaSFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       t_alpha=t_alpha,
                       )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'magface':
        head = MagFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'utilface':
        head = utilFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
 
    else:
        raise ValueError('not a correct head type', head_type)
    return head

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

####TODO 

class utilFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(utilFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))

        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('classwise_mean', torch.ones(classnum)*(20))
        self.register_buffer('batch_std', torch.ones(1)*(10))
        self.soft_max = nn.Softmax(dim=1)
        # self.soft_max = nn.Softmax()
        self.register_buffer('distill_buffer', self.soft_max(torch.ones(classnum,classnum)*0.5))
        self.kl_loss = nn.KLDivLoss(reduction="none")
        
        self.linear = torch.nn.Linear(embedding_size, 1)
        

        ### KL sampling 
        self.num_sample = self.classnum * 0.01 # about 100 ~ 


        print('\n\CWFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)



    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        
        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()
        

        distill_cos = (cosine.detach() + 1 )/2 
        distill_cos = self.soft_max(distill_cos)

        # update batchmean batchstd
        with torch.no_grad():

            lu = label.unique()
            # ## group by mean! 
            M = torch.zeros(self.classnum,label.size()[0]).cuda()
            M[label.squeeze(),torch.arange(label.size()[0])] = 1
            M = torch.nn.functional.normalize(M, p=1, dim=1)
            
            batch_group_mean = torch.mm(M,safe_norms).squeeze().detach()
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
                        
            self.classwise_mean[lu] = batch_group_mean[lu] * self.t_alpha + (1 - self.t_alpha) * self.classwise_mean[lu]
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        
            ## group logit? TODO add weight? 
            batch_logit_mean = torch.mm(M,distill_cos).squeeze()
            
            # self.distill_buffer[lu] = batch_logit_mean[lu] * self.t_alpha + (1 - self.t_alpha) * self.distill_buffer[lu]
            
            ## topk with cosine value
            ## TODO is random value with origin label

            # cos = distill_cos.clone().detach()
            # cos[torch.arange(label.size()[0]), label.squeeze()] = 2.0
            # distill_idx = torch.topk(cos, k=int(self.num_sample))[1]
            uniform = torch.ones((label.size()[0])).cuda()
            uniform = uniform/uniform.sum()
            
            

        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        # norm_weight = (safe_norms - self.batch_mean) / (self.batch_std+self.eps)
        # norm_weight = torch.clip(norm_weight * self.h , -1, 1)
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333 not for classwize 
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        ##class-wize static 
        cw_margin_scaler = (safe_norms - self.classwise_mean[label].unsqueeze(1)) / (self.batch_std+self.eps) # 66% between -1, 1
        cw_margin_scaler = torch.clip(cw_margin_scaler, -1, 1) * self.h

        cw_cosine = cosine.clone()
        
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        # g_angular = self.m * 1
        # import pdb ; pdb.set_trace()
        origin_m_arc = m_arc * g_angular
        theta = cosine.acos()

        theta_m = torch.clip(theta + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        ## CW
        g_angular_cw = self.m * cw_margin_scaler * -1
        cw_m_arc = m_arc * g_angular_cw
        cw_theta = cw_cosine.acos()

        cw_theta_m = torch.clip(cw_theta + cw_m_arc, min=self.eps, max=math.pi-self.eps)
        cw_cosine = cw_theta_m.cos()
        
        # print("first",(cosine == cw_cosine).sum()/cw_cosine.reshape(-1).shape[0])

        ######


        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        origin_m_cos = m_cos * g_add
        cosine = cosine - origin_m_cos

        # ## CW 
        cw_g_add = self.m + (self.m * cw_margin_scaler)
        cw_m_cos = m_cos * cw_g_add
        cw_cosine = cw_cosine - cw_m_cos




        # scale
        scaled_cosine_m = cosine * self.s
        scaled_cw_cosine_m = cw_cosine * self.s

        # print("last",(scaled_cosine_m == scaled_cw_cosine_m).sum()/scaled_cosine_m.reshape(-1).shape[0])

        ## distilation  
        # import pdb ; pdb.set_trace()

        # origin 
        # mean_kl_loss = (((1-margin_scaler)/2)*self.kl_loss(distill_cos,self.distill_buffer[label.squeeze()])).mean()
        ## select random idx
         
        # mean_kl_loss = self.kl_loss(self.soft_max(torch.gather(distill_cos, 1, distill_idx)),self.soft_max(torch.gather(self.distill_buffer[label.squeeze()], 1, distill_idx)))
        

        mean_kl_loss = torch.where(safe_norms<=self.batch_mean,self.kl_loss(safe_norms.squeeze()/safe_norms.sum(),uniform),torch.tensor(0, dtype=safe_norms.dtype).to(safe_norms.device)).mean()
        # print(mean_kl_loss.shape)
        # mean_kl_loss = self.kl_loss(self.soft_max(safe_norms.squeeze()),uniform).mean()

        # return scaled_cosine_m, norm_weight, mean_kl_loss
        return scaled_cosine_m, cw_margin_scaler, mean_kl_loss, scaled_cw_cosine_m
        # return scaled_cosine_m, cw_margin_scaler, None, None



## nomalize norm out 
class AdaSFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaSFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        self.kernel2 = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.kernel2.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)


    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        kernel_norm2 = l2_norm(self.kernel2,axis=0)
        cosine2 = torch.mm(embbedings,kernel_norm2)
        cosine2 = cosine2.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)

        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        
        
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        ### for tail 
        theta2 = cosine2.acos()
        theta_m2 = torch.clip(theta2 + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine2 = theta_m2.cos()
        
        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        ### for tail 
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine2 = cosine2 - m_cos



        # scale
        scaled_cosine_m = cosine * self.s
        scaled_cosine_2 = cosine2 * self.s

        return scaled_cosine_m, scaled_cosine_2 , margin_scaler


####TODO 

## nomalize norm out 
class AdaWIndexFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 
                 ):
        super(AdaWIndexFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        # self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        # self.register_buffer('batch_std', torch.ones(1)*100)
        self.register_buffer('batch_std', torch.ones(1)*100)


    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)

        g_angular = self.m * margin_scaler * -1
        


        m_arc = m_arc * g_angular
        
        
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        ### MAX INDEX
        argmax_idx = torch.argmax(cosine,axis=1)
        max_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        max_arc.scatter_(1, argmax_idx.reshape(-1, 1), 1.0)
        
        max_arc = max_arc * g_angular

        theta = cosine.acos()
        theta_max = torch.clip(theta + max_arc, min=self.eps, max=math.pi-self.eps)
        cosine_max = theta_max.cos()
        
        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        ### MAX INDEX 
        max_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        max_cos.scatter_(1, argmax_idx.reshape(-1, 1), 1.0)

        max_cos = max_cos * g_add
        cosine_max = cosine_max - max_cos

        # scale
        scaled_cosine_m = cosine * self.s
        scaled_cosine_max = cosine_max * self.s

        return scaled_cosine_m, scaled_cosine_max, argmax_idx, margin_scaler

    def get_weight_norm(self):
        
        return torch.norm(self.kernel,2,0)



class LAMAFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(LAMAFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        
        self.kernel_mb = torch.zeros(embedding_size,classnum)

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        #self.register_buffer('batch_mean', torch.ones(1)*(20))
        #self.register_buffer('batch_mean', torch.ones(classnum)*(20))
        #self.register_buffer('batch_std', torch.ones(1)*(10))
        
        #TODO
        self.feature_mb = [torch.zeros(0).cuda()]*classnum
        self.proxy_mb = [torch.zeros(0).cuda()]*classnum
        
        #self.register_buffer('feature_mb', [torch.zeros(0).cuda()]*classnum)
        #self.register_buffer('proxy_mb',[torch.zeros(0).cuda()]*classnum)
        
        self.queue_size = 30  ## hyperparm... 
        self.compensate = False ## important 
          
       
        print('\n\CWCFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def update(self, feature_norm, label, kernel_norm):
        
        for fn,lb in zip(feature_norm,label):
            self.feature_mb[lb] = torch.cat([self.feature_mb[lb],fn.data],dim=0)
            self.proxy_mb[lb] = torch.cat([self.proxy_mb[lb],kernel_norm[lb].unsqueeze(0).data],dim=0)
            
        
        for lu in label.unique():
            over_size = self.feature_mb[lu].shape[0] - self.queue_size
            if over_size > 0:
                self.feature_mb[lu] = self.feature_mb[lu][over_size:]
                self.proxy_mb[lu] = self.proxy_mb[lu][over_size:]
        
        
    def feature_norm_normalize(self,feature_norm,label):
        ## normalize by estimated mean and std
       
        kernel_norm = torch.norm(self.kernel,dim=0)
        
        self.update(feature_norm,label,kernel_norm)
        
        res = torch.zeros(feature_norm.shape).cuda()

        for val,(fn,lb) in enumerate(zip(feature_norm,label)):
            if self.compensate:
                ## fail
                compensate_val = self.feature_mb[lb] * kernel_norm[lb] / (self.proxy_mb[lb]+1e-3)
                
                ## ver2
                ##compensate_val = self.feature_mb[lb] + (self.feature_mb[lb]/self.proxy_mb[lb]) * (kernel_norm[lb] - self.proxy_mb[lb])
            else:
                compensate_val = self.feature_mb[lb]
            
            
            if compensate_val.shape[0]>2:
                res[val] = (fn - compensate_val.mean()) / (compensate_val.std() + self.eps)
            else:
                res[val] = (fn - compensate_val.mean()) / 20 



        return res




class CWCFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(CWCFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        
        self.kernel_mb = torch.zeros(embedding_size,classnum)

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        #self.register_buffer('batch_mean', torch.ones(1)*(20))
        #self.register_buffer('batch_mean', torch.ones(classnum)*(20))
        #self.register_buffer('batch_std', torch.ones(1)*(10))
        
        #TODO
        self.feature_mb = [torch.zeros(0).cuda()]*classnum
        self.proxy_mb = [torch.zeros(0).cuda()]*classnum
        
        #self.register_buffer('feature_mb', [torch.zeros(0).cuda()]*classnum)
        #self.register_buffer('proxy_mb',[torch.zeros(0).cuda()]*classnum)
        
        self.queue_size = 30  ## hyperparm... 
        self.compensate = False ## important 
          
       
        print('\n\CWCFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def update(self, feature_norm, label, kernel_norm):
        
        for fn,lb in zip(feature_norm,label):
            self.feature_mb[lb] = torch.cat([self.feature_mb[lb],fn.data],dim=0)
            self.proxy_mb[lb] = torch.cat([self.proxy_mb[lb],kernel_norm[lb].unsqueeze(0).data],dim=0)
            
        
        for lu in label.unique():
            over_size = self.feature_mb[lu].shape[0] - self.queue_size
            if over_size > 0:
                self.feature_mb[lu] = self.feature_mb[lu][over_size:]
                self.proxy_mb[lu] = self.proxy_mb[lu][over_size:]
        
        
    def feature_norm_normalize(self,feature_norm,label):
        ## normalize by estimated mean and std
       
        kernel_norm = torch.norm(self.kernel,dim=0)
        
        self.update(feature_norm,label,kernel_norm)
        
        res = torch.zeros(feature_norm.shape).cuda()

        for val,(fn,lb) in enumerate(zip(feature_norm,label)):
            if self.compensate:
                ## fail
                compensate_val = self.feature_mb[lb] * kernel_norm[lb] / (self.proxy_mb[lb]+1e-3)
                
                ## ver2
                ##compensate_val = self.feature_mb[lb] + (self.feature_mb[lb]/self.proxy_mb[lb]) * (kernel_norm[lb] - self.proxy_mb[lb])
            else:
                compensate_val = self.feature_mb[lb]
            
            
            if compensate_val.shape[0]>2:
                res[val] = (fn - compensate_val.mean()) / (compensate_val.std() + self.eps)
            else:
                res[val] = (fn - compensate_val.mean()) / 20 



        return res

        
    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability
                
        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        
        with torch.no_grad():
            margin_scaler = self.feature_norm_normalize(safe_norms,label) 
            margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
            margin_scaler = torch.clip(margin_scaler, -1, 1)
    

        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
                
        
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        #512 clsnum
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m




class CWLMAGFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 datamod = None
                 ):
        super(CWLMAGFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        
        self.datamod = datamod
        
        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)
    
    def G_loss(self,norm,z=5):
        ## norm [-1,1]
        ## -> 0 to 110 

        #return torch.exp(-z*norm)/np.exp(z)/10
        return torch.exp(-norm/z)/10

    def forward(self, embbedings, norms, label):
        
        
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()         
        
        kernel_mag = torch.norm(self.kernel,dim=0)[label]
        kernel_mag = kernel_mag.clone().detach()
        kernel_mag = torch.clip(kernel_mag,min=kernel_mag.mean()-2*kernel_mag.std())
        safe_norms = safe_norms / (kernel_mag.unsqueeze(1) + 1e-3)
        
        safe_norms = torch.clip(safe_norms, min=0.001, max=100) # for stability
        
        loss_norms = torch.clip(norms, min=0.001, max=110).clone() # for stability


        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        
        g_angular = 0.6+ 0.2*margin_scaler * -1
        
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
#         m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
#         m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
#         g_add = self.m + (self.m * margin_scaler)
#         m_cos = m_cos * g_add
#         cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        
        #return scaled_cosine_m, self.G_loss(margin_scaler)
        return scaled_cosine_m, self.G_loss(loss_norms)


class CWLNFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 datamod = None
                 ):
        super(CWLNFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        
        self.datamod = datamod
        
        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label,class_sample_num_):
        
        
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        #safe_norms = norms
        
        ## 
#         kernel_mag = torch.norm(self.kernel,dim=0)[label]
#         kernel_mag = kernel_mag.clone().detach()
        safe_norms = safe_norms.clone().detach() 
        #print("kernel_mag",kernel_mag.shape)
        #print("safe_norms",safe_norms.shape)
        safe_norms = safe_norms / (class_sample_num_.unsqueeze(1)+1e-3)
        #safe_norms = safe_norms / (kernel_mag.unsqueeze(1) + 1)
        
#         print('class',class_sample_num_)        
        safe_norms = torch.clip(safe_norms, min=0.001, max=100) # for stability
        
        #print("safe_norms",safe_norms.shape)

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        
        #print('m_arc',m_arc.size())
        #512 10572

        g_angular = self.m * margin_scaler * -1
        
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m



class CWLFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(CWLFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(10))
        self.register_buffer('batch_std', torch.ones(1)*10)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        #safe_norms = norms
        
        ## 
        kernel_mag = torch.norm(self.kernel,dim=0)[label]
        kernel_mag = kernel_mag.clone().detach()
        
        kernel_mag = torch.clip(kernel_mag,min=kernel_mag.mean()-2*kernel_mag.std())
        
        safe_norms = safe_norms.clone().detach() 
        #print("kernel_mag",kernel_mag.shape)
        #print("safe_norms",safe_norms.shape)
        safe_norms = safe_norms / (kernel_mag.unsqueeze(1) + 1e-5)
        #safe_norms = safe_norms / (kernel_mag.unsqueeze(1) + 1)
        
        safe_norms = torch.clip(safe_norms, min=0.001, max=100) # for stability
        
        #print("safe_norms",safe_norms.shape)

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        
        #print('m_arc',m_arc.size())
        #512 10572

        
        g_angular = self.m * margin_scaler * -1
        
        #print('g_ang',g_angular.size())
        ## 512 1

        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class CWFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(CWFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        #self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_mean', torch.ones(classnum)*(20))
        self.register_buffer('batch_std', torch.ones(1)*(10))
        
        # for class weight 
        #self.tc_alpha = t_alpha
        #self.register_buffer('tc',torch.zeros(1))
        #self.register_buffer('cb_mean',torch.ones(1)*(0.7))
        #self.register_buffer('cb_std',torch.ones(1)*1)


        print('\n\CWFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability
        
        #print('cos',cosine.size())
        #512,clsnum

        
        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        
       
        #kernel_mag = torch.norm(self.kernel,dim=0)[label]
        #kernel_mag = kernel_mag.clone().detach()
        #tmp_kernel_mag = torch.clip(kernel_mag,min=kernel_mag.mean()-2*kernel_mag.std())
        
        #safe_norms = safe_norms.clone().detach() 
        #safe_norms = safe_norms / (tmp_kernel_mag.unsqueeze(1) + 1e-5)
        
        #safe_norms = torch.clip(safe_norms, min=0.001, max=100)
        

        # update batchmean batchstd
        with torch.no_grad():

            lu = label.unique()
            ## group by mean! 
            M = torch.zeros(self.classnum,label.size()[0]).cuda()
            M[label.squeeze(),torch.arange(label.size()[0])] = 1
            M = torch.nn.functional.normalize(M, p=1, dim=1)
            
            #print("M",M.size())
            #print("safe_norms",safe_norms.size())
            batch_group_mean = torch.mm(M,safe_norms).squeeze().detach()
            
            #print("batch_group",batch_group_mean.size())
            #print("self.bath_mean",self.batch_mean.size())

            #mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            
            #self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            
            self.batch_mean[lu] = batch_group_mean[lu] * self.t_alpha + (1 - self.t_alpha) * self.batch_mean[lu]
            
            #std = torch.sqrt((self.batch_mean[label.squeeze()] - safe_norms.squeeze())**2 / label.size()[0])
            
            #print("#############################################")
            #print("label",label.size())
            #print("batch_mean",self.batch_mean[label.squeeze()].size())
            #print("safe_nomrs",safe_norms.size())
            ## 512 1 
            #print("std",std.size())
            
            #batch_group_std = torch.mm(M,std.unsqueeze(1)).squeeze().detach()

            

            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
            #self.batch_std[lu] =  batch_group_std[lu] * self.t_alpha + (1 - self.t_alpha) * self.batch_std[lu]
            
            ## TODO do we need clip the mean too...?  
            #tmp_batch_std = torch.clip(self.batch_std,min=self.batch_std.mean()-2*self.batch_std.std())
            #tmp_batch_std = self.batch_std
            #tmp_batch_mean = torch.clip(self.batch_mean,min=self.batch_mean.mean()-2*self.batch_mean.std())


            #print(self.batch_mean)
            
            

        
        #TODO 2..? 
        #class_weight_norm = torch.norm(self.kernel,2,dim=0)
        #safe_kernel_norms = torch.clip(class_weight_norm, min = 0.001,max=2)
        #safe_kernel_norms = safe_kernel_norms.clone().detach()

        #update kernel weight norm mean, std
        #with torch.no_grad():
            #mean = safe_kernel_norms.mean().detach()
            #std = safe_kernel_norms.std().detach()
            #self.cb_mean = mean * self.tc_alpha + (1 - self.tc_alpha) * self.cb_mean
            #self.cb_std =  std * self.tc_alpha + (1 - self.tc_alpha) * self.cb_std
        
        
        #print('safe_norm',safe_norms.size())
        #512 1
        #print('btm',self.batch_mean[label].size())
        #print('bts',self.batch_std[label].size())

        #margin_scaler = (safe_norms - self.batch_mean[label].unsqueeze(1)) / (self.batch_std[label].unsqueeze(1)+self.eps) # 66% between -1, 1
        margin_scaler = (safe_norms - self.batch_mean[label].unsqueeze(1)) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
    
        

        #print(f"MS {margin_scaler.size()}")
        #512 512 

        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        

        ###### class
        
        #class_scaler = (safe_kernel_norms - self.cb_mean) / (self.cb_std + self.eps)
        #class_scaler = class_scaler * self.h 
        #class_scaler = torch.clip(class_scaler,-1,1)

        ######

        # g_angular

        ###
        #label_cs = -class_scaler[label].unsqueeze(1)
        ###
        
        #print("margin:",margin_scaler.shape)
        #print("lable:",label_cs.shape)
        
        
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        #512 clsnum
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        #g_angular = self.m * (margin_scaler + label_cs) * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        #g_add = self.m + (self.m * (margin_scaler + label_cs))
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

class MagFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 l_margin = 0.45,
                 u_margin = 0.8,
                 l_a = 10,
                 u_a = 110,
                 ):
        super(MagFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a

    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)


        ada_margin = self._margin(norms)        
        m_arc = m_arc * ada_margin
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()


        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
        






class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(10))
        self.register_buffer('batch_std', torch.ones(1)*10)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        
        #print('margin_s',margin_scaler.size())
        #512 1
        

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        
        #print('m_arc',m_arc.size())
        #512 10572

        g_angular = self.m * margin_scaler * -1
        
        
        
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

    def get_weight_norm(self):
        
        return torch.norm(self.kernel,2,0)




class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace(Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369

        self.eps = 1e-4

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m
