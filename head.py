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
    elif head_type == 'fraface':
        head = fraFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'fratface':
        head = fratFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'qualface':
        head = qualFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'ufoface':
        head = UfoFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'arcface':
        head = ArcFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'arcwface':
        head = ArcWFace(embedding_size=embedding_size,
                classnum=class_num,
                m=m,
                s=s,
                )
    elif head_type == 'coswface':
        head = CoswFace(embedding_size=embedding_size,
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

def remove_common_elements(target, ref):
    target_sorted = target.sort().values
    ref_sorted = ref.sort().values
    matches = (target_sorted.unsqueeze(1) != ref_sorted.unsqueeze(0)).nonzero()
    fin = target_sorted[matches[:, 0]]
    # Removing potential duplicate matches
    return torch.unique(fin)


class CoswFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.4, t_alpha=1.0, h=0.333):
        super(CoswFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4
        self.h = h

        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(10))
        self.register_buffer('batch_std', torch.ones(1)*10)

        self.ood_feature_mem = torch.zeros(0).cuda()
        self.ood_feature_time = torch.zeros(0).cuda()
        self.queue_size = 300  #hyper-param


        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)
    def update(self, feature, norm):
        
        ## feature tensor 1..., embedding
        self.ood_feature_mem = torch.cat([feature,self.ood_feature_mem],dim=0)

        self.ood_feature_time =   self.ood_feature_time + 1 
        self.ood_feature_time = torch.cat([torch.ones(feature.shape[0]).cuda(),self.ood_feature_time],dim=0)

        over_size = self.ood_feature_mem.shape[0] - self.queue_size
        if over_size > 0:
            self.ood_feature_mem = self.ood_feature_mem[over_size:]
            self.ood_feature_time = self.ood_feature_time[over_size:]
        
        time_idx = torch.where(self.ood_feature_time>1000,False,True)##hyperparam

        self.ood_feature_mem = self.ood_feature_mem[time_idx]
        self.ood_feature_time = self.ood_feature_time[time_idx]

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()


        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        
        with torch.no_grad():   

            max_cos = torch.max(cosine,dim=1,keepdim=True)[0]
            cond = torch.where((max_cos<0.5)&(margin_scaler>0.2),True,False)
            self.update(embbedings[cond.squeeze()],norms[cond.squeeze()])
     
        if self.ood_feature_mem.shape[0] >0:
            OOD_sep = torch.mm(embbedings,self.ood_feature_mem.T)
            OOD_sep = OOD_sep.clamp(-1+self.eps, 1-self.eps)
            OOD_sep = OOD_sep.unsqueeze(2)
            ood_feature_mem_expanded = self.ood_feature_mem.unsqueeze(0)
            OOD_proj = OOD_sep * ood_feature_mem_expanded
            OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2)[~cond.squeeze()].mean()
            # OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2).mean()
        else:
            OOD_proj_loss = 0

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m,cond, OOD_proj_loss,margin_scaler



class ArcWFace(Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5,t_alpha=1.0,h=0.333):
        super(ArcWFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.h = h
        self.eps = 1e-3

        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(10))
        self.register_buffer('batch_std', torch.ones(1)*10)

        self.ood_feature_mem = torch.zeros(0).cuda()
        # self.ood_feature_norm = torch.zeros(0).cuda()
        self.ood_feature_time = torch.zeros(0).cuda()
        self.queue_size = 300  #hyper-param


    def update(self, feature, norm):
        
        ## feature tensor 1..., embedding
        self.ood_feature_mem = torch.cat([feature,self.ood_feature_mem],dim=0)
        # self.ood_feature_norm = torch.cat([feature,self.ood_feature_norm],dim=0)

        self.ood_feature_time =   self.ood_feature_time + 1 
        self.ood_feature_time = torch.cat([torch.ones(feature.shape[0]).cuda(),self.ood_feature_time],dim=0)

        over_size = self.ood_feature_mem.shape[0] - self.queue_size
        if over_size > 0:
            self.ood_feature_mem = self.ood_feature_mem[over_size:]
            # self.ood_feature_norm = self.ood_feature_norm[over_size:]
            self.ood_feature_time = self.ood_feature_time[over_size:]
        
        time_idx = torch.where(self.ood_feature_time>1000,False,True)##hyperparam

        self.ood_feature_mem = self.ood_feature_mem[time_idx]
        self.ood_feature_time = self.ood_feature_time[time_idx]
        # self.ood_feature_norm = self.ood_feature_norm[time_idx]


    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        
        with torch.no_grad():   

            max_cos = torch.max(cosine,dim=1,keepdim=True)[0]
            cond = torch.where((max_cos<0.5)&(margin_scaler>0.2),True,False)
            self.update(embbedings[cond.squeeze()],norms[cond.squeeze()])
     
        if self.ood_feature_mem.shape[0] >0:
            OOD_sep = torch.mm(embbedings,self.ood_feature_mem.T)
            OOD_sep = OOD_sep.clamp(-1+self.eps, 1-self.eps)
            OOD_sep = OOD_sep.unsqueeze(2)
            ood_feature_mem_expanded = self.ood_feature_mem.unsqueeze(0)
            OOD_proj = OOD_sep * ood_feature_mem_expanded
            # OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2)[~cond.squeeze()].mean()
            OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2).mean()
        else:
            OOD_proj_loss = 0


        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m,cond, OOD_proj_loss,margin_scaler


class UfoFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(UfoFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # self.proj_head = Parameter(torch.Tensor(embedding_size,embedding_size))
        # self.proj_head2 = Parameter(torch.Tensor(embedding_size,embedding_size))
        # self.relu = torch.nn.ReLU()

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        # self.proj_head.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        # self.proj_head2.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(10))
        self.register_buffer('batch_std', torch.ones(1)*10)

        self.ood_feature_mem = torch.zeros(0).cuda()
        # self.ood_feature_norm = torch.zeros(0).cuda()
        self.ood_feature_time = torch.zeros(0).cuda()
        self.queue_size = 300  #hyper-param

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def update(self, feature, norm):
        
        ## feature tensor 1..., embedding
        self.ood_feature_mem = torch.cat([feature,self.ood_feature_mem],dim=0)
        # self.ood_feature_norm = torch.cat([feature,self.ood_feature_norm],dim=0)

        self.ood_feature_time =   self.ood_feature_time + 1 
        self.ood_feature_time = torch.cat([torch.ones(feature.shape[0]).cuda(),self.ood_feature_time],dim=0)

        over_size = self.ood_feature_mem.shape[0] - self.queue_size
        if over_size > 0:
            self.ood_feature_mem = self.ood_feature_mem[over_size:]
            # self.ood_feature_norm = self.ood_feature_norm[over_size:]
            self.ood_feature_time = self.ood_feature_time[over_size:]
        
        time_idx = torch.where(self.ood_feature_time>1000,False,True)##hyperparam

        self.ood_feature_mem = self.ood_feature_mem[time_idx]
        self.ood_feature_time = self.ood_feature_time[time_idx]
        # self.ood_feature_norm = self.ood_feature_norm[time_idx]


    def forward(self, embbedings, norms, label):
        # print(label.shape)
        # print(self.classnum)
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
        
        with torch.no_grad():   
            ## check OOD and 
            # generate mask for OOD sample ?????

            max_cos = torch.max(cosine,dim=1,keepdim=True)[0]
            # print(max_cos.shape)
            cond = torch.where((max_cos<0.5)&(margin_scaler>0.2),True,False)
            # print("emb cond",embbedings[cond.squeeze()].shape)
            self.update(embbedings[cond.squeeze()],norms[cond.squeeze()])

        ##TODO
        # print(self.ood_feature_mem.shape[0])

        # proj=(torch.mm(self.relu(embbedings),self.proj_head))
        # proj_t=(torch.mm(self.relu(embbedings*norms),self.proj_head))
        # proj=(torch.mm(self.relu(proj_t),self.proj_head2))

        if self.ood_feature_mem.shape[0] >0:
            # print("ood_shape",self.ood_feature_mem.shape)

            # batch, 300 
            # Q,R = torch.linalg.qr(self.ood_feature_mem.T)
            # OOD_sep = torch.mm(embbedings*norms,Q.T)
            OOD_sep = torch.mm(embbedings,self.ood_feature_mem.T)
            OOD_sep = OOD_sep.clamp(-1+self.eps, 1-self.eps)
            OOD_sep = OOD_sep.unsqueeze(2)
            ood_feature_mem_expanded = self.ood_feature_mem.unsqueeze(0)

            # 512 300 512
            OOD_proj = OOD_sep * ood_feature_mem_expanded

            # 512 300 512 
            # wo_proj_direction = (embbedings*norms).unsqueeze(1) - OOD_proj
            # print(Q.shape)
            # print(embbedings.shape)
            # wo_proj_direction = (embbedings*norms - (embbedings*norms)@Q@Q.T).clone().detach()

            OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2)[~cond.squeeze()].mean()
            # OOD_proj_loss = torch.sum(torch.abs(OOD_proj),dim=2).mean()
            # print(torch.sum(OOD_proj - proj,dim=2))
            # OOD_proj_loss = (wo_proj_direction - proj)**2

            # print(OOD_proj_loss.shape)

        else:
            OOD_proj_loss = 0
            


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
        return scaled_cosine_m, cond, OOD_proj_loss,margin_scaler



### pred quality utilize feature norm and long-tail
class qualFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(qualFace, self).__init__()
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
        
        self.qual_linear = torch.nn.Linear(embedding_size, 1)
        self.dist_linear = torch.nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        

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
        
        distill_cos = cosine.clone()
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

        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333 not for classwize 
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        ##class-wize static 
        cw_margin_scaler = (safe_norms - self.classwise_mean[label].unsqueeze(1)) / (self.batch_std+self.eps) # 66% between -1, 1
        cw_margin_scaler = torch.clip(cw_margin_scaler, -1, 1) * self.h


        # qality_est = self.qual_linear(embbedings)

        qality_cos = cosine.clone().detach()

        mask = torch.ones_like(qality_cos, dtype=torch.bool)
        mask[torch.arange(qality_cos.shape[0]), label] = False
        masked_tensor = torch.where(mask, qality_cos, torch.tensor(-1,dtype=qality_cos.dtype).to(qality_cos.device))
        max_value_indices = torch.argmax(masked_tensor, dim=1)

        # q_est_loss = (qality_est - ((cw_margin_scaler+1))* (qality_cos[torch.arange(label.size()[0]),label]/(qality_cos[torch.arange(label.size()[0]),max_value_indices]+self.eps)))**2


        qality_est = self.qual_linear(embbedings)
        dist_est = self.dist_linear(embbedings)

        q_est_loss = (self.sigmoid(qality_est) - ((cw_margin_scaler+1)/2))**2
        # dist_est_loss = (self.sigmoid(dist_est)- ((((margin_scaler+1)/2)/((cw_margin_scaler+2)))))**2
        dist_est_loss = (self.sigmoid(dist_est) - qality_cos[torch.arange(label.size()[0]),label]/(qality_cos[torch.arange(label.size()[0]),max_value_indices]+self.eps))**2

        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        origin_m_arc = m_arc * g_angular
        theta = cosine.acos()

        theta_m = torch.clip(theta + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        ######
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        origin_m_cos = m_cos * g_add
        cosine = cosine - origin_m_cos

        

        # scale
        scaled_cosine_m = cosine * self.s


        return scaled_cosine_m, q_est_loss, dist_est_loss


### freeze version, add weight dim 
class fratFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(fratFace, self).__init__()
        self.classnum = classnum
        ## 0 normal 1 class specific
        self.kernel = Parameter(torch.Tensor(embedding_size,2,classnum))

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

        ### sampling 
        self.num_sample = int(self.classnum*0.2)  


        print('\n fraFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)



    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        # cosine = torch.mm(embbedings,kernel_norm)

        # print(embbedings.shape)
        # print(kernel_norm.shape)
        # import pdb ; pdb.set_trace()

        # cosine = torch.matmul(embbedings,kernel_norm)
        cosine = torch.einsum('be,etc->btc',embbedings,kernel_norm)

        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # mask = torch.ones_like(self.kernel, dtype=torch.bool,device=self.kernel.device)
        # mask[:,1,:] = False
        # mask[torch.arange(label.size()[0]),1,label.squeeze()] = True
        # hook_ = self.kernel.register_hook(lambda grad: grad * mask.float())

        # update batchmean batchstd
        with torch.no_grad():
            
            lu = label.unique()
            # for idx in freaze_idx:
            #     self.kernel[:,idx].requires_grad = False
            #     print(self.kernel[:,idx].requires_grad)

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
        
            ### sampling...? 

            # cos_cond = (cosine[torch.arange(label.size()[0]),label]>0.5).unsqueeze(1).expand_as(distill_cos)
            # cond = torch.where(cos_cond,distill_cos.detach(),torch.tensor(0, dtype=distill_cos.dtype).to(distill_cos.device))


        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        # norm_weight = (safe_norms - self.batch_mean) / (self.batch_std+self.eps)
        # norm_weight = torch.clip(norm_weight * self.h , -1, 1)
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333 not for classwize 
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        ##class-wize static 
        cw_margin_scaler = (safe_norms - self.classwise_mean[label].unsqueeze(1)) / (self.batch_std+self.eps) # 66% between -1, 1
        cw_margin_scaler = torch.clip(cw_margin_scaler, -1, 1) * self.h

        w1_cosine = cosine[:,0,:]
        # B C  
        w2_cosine = cosine[:,0,:].clone()
        w2_cosine[torch.arange(label.size()[0]),label.squeeze()] = cosine[torch.arange(label.size()[0]),1,label.squeeze()]

        mask = torch.zeros_like(w2_cosine, dtype=torch.bool,device=w2_cosine.device)
        mask[torch.arange(label.size()[0]),label.squeeze()] = True
        hook_ = w2_cosine.register_hook(lambda grad: grad * mask)


        ## W1 #####################################
        m_arc = torch.zeros(label.size()[0], w1_cosine.size()[1], device=w1_cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        

        # import pdb ; pdb.set_trace()
        origin_m_arc = m_arc * g_angular
        theta = w1_cosine.acos()

        theta_m = torch.clip(theta + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        w1_cosine = theta_m.cos()


        m_cos = torch.zeros(label.size()[0], w1_cosine.size()[1], device=w1_cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        origin_m_cos = m_cos * g_add
        w1_cosine = w1_cosine - origin_m_cos

        # scale
        scaled_cosine_m = w1_cosine * self.s


        ## W2 ##########################################
        theta2 = w2_cosine.acos()
        theta_m2 = torch.clip(theta2 + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        w2_cosine = theta_m2.cos()

        w2_cosine = w2_cosine - origin_m_cos

        # scale
        scaled_cosine_m2 = w2_cosine * self.s


        return scaled_cosine_m,scaled_cosine_m2, cw_margin_scaler ,hook_


### freeze version
class fraFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(fraFace, self).__init__()
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

        ### sampling 
        self.num_sample = int(self.classnum*0.2)  


        print('\n fraFace with the following property')
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

            random_indices = torch.randperm(self.classnum)[:self.num_sample].cuda()
            lu = label.unique()
            freaze_idx = remove_common_elements(random_indices,lu)
            # for idx in freaze_idx:
            #     self.kernel[:,idx].requires_grad = False
            #     print(self.kernel[:,idx].requires_grad)




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
        
        
      
        #TODO consider batch 
        mask = torch.ones_like(self.kernel, dtype=torch.bool,device=self.kernel.device)
        mask[:,freaze_idx] = False
        hook_ = self.kernel.register_hook(lambda grad: grad * mask.float())
        # print(self.kernel.grad)

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

        # import pdb ; pdb.set_trace()
        origin_m_arc = m_arc * g_angular
        theta = cosine.acos()

        theta_m = torch.clip(theta + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()


        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        origin_m_cos = m_cos * g_add
        cosine = cosine - origin_m_cos


        # scale
        scaled_cosine_m = cosine * self.s

        return scaled_cosine_m,hook_



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
        
        distill_cos = cosine.clone()
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

        
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        cw_margin_scaler = (safe_norms - self.classwise_mean[label].unsqueeze(1)) / (self.batch_std+self.eps) # 66% between -1, 1
        ### weight for distill buffer 
        with torch.no_grad():
            # import pdb ; pdb.set_trace()
            cos_cond = (cosine[torch.arange(label.size()[0]),label]>0.8).unsqueeze(1).expand_as(distill_cos)
            cond = torch.where(cos_cond,distill_cos.detach(),torch.tensor(0, dtype=distill_cos.dtype).to(distill_cos.device))
            batch_logit_mean = torch.mm(M,cond).squeeze()
            self.distill_buffer[lu] = batch_logit_mean[lu] * self.t_alpha + (1 - self.t_alpha) * self.distill_buffer[lu]


        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333 not for classwize 
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        ##class-wize static 
        cw_margin_scaler = torch.clip(cw_margin_scaler, -1, 1) * self.h

        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        origin_m_arc = m_arc * g_angular
        theta = cosine.acos()

        theta_m = torch.clip(theta + origin_m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)

        g_add = self.m + (self.m * margin_scaler)
        origin_m_cos = m_cos * g_add
        cosine = cosine - origin_m_cos



        # scale
        scaled_cosine_m = cosine * self.s

        # origin 
        # mean_kl_loss = (((1-margin_scaler)/2)*self.kl_loss(distill_cos,self.distill_buffer[label.squeeze()]/cw_margin_scaler)).mean()
        mean_kl_loss = self.kl_loss(distill_cos,self.distill_buffer[label.squeeze()]/(2-margin_scaler)).mean()

        ## select random idx

        return scaled_cosine_m, cw_margin_scaler, mean_kl_loss, None



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
