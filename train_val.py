import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
import evaluate_utils
import head
import net
import numpy as np
import utils
import pickle


class Trainer(LightningModule):
    def __init__(self,**kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams

        self.class_num = utils.get_num_class(self.hparams)
        print('classnum: {}'.format(self.class_num))
               

        self.model = net.build_model(model_name=self.hparams.arch)
        
        self.head = head.build_head(head_type=self.hparams.head,
                                     embedding_size=512,
                                     class_num=self.class_num,
                                     m=self.hparams.m,
                                     h=self.hparams.h,
                                     t_alpha=self.hparams.t_alpha,
                                     s=self.hparams.s,
                                     )

        self.cross_entropy_loss = CrossEntropyLoss()
        self.cross_entropy_loss_non_redu = CrossEntropyLoss(reduction='none')
        self.class_sample_num = self.hparams.head=='cwlnface'
        self.relu = torch.nn.ReLU()

        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            self.model.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})


    def forward(self, images, labels,class_sample_num_=None):
        embeddings, norms = self.model(images)
        if self.class_sample_num:
            cos_thetas = self.head(embeddings, norms, labels,class_sample_num_)
        else:
            if self.hparams.head == 'cwlmagface':
                cos_thetas,normalized_norm = self.head(embeddings, norms, labels)
            else:
                cos_thetas = self.head(embeddings, norms, labels)
        
        if isinstance(cos_thetas, tuple):
            print("cos_thetas",cos_thetas)
            cos_thetas, bad_grad = cos_thetas
            print(bad_grad)
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
            
        if self.hparams.head == 'cwlmagface':
             return cos_thetas, normalized_norm, embeddings, labels
        else:
            return cos_thetas, norms, embeddings, labels


    def training_step(self, batch, batch_idx):
        
        if self.class_sample_num:
            images, (labels,class_sample_num_) = batch
            cos_thetas, norms, embeddings, labels = self.forward(images, labels,class_sample_num_)
        else:
            images, labels = batch
            cos_thetas, norms, embeddings, labels = self.forward(images, labels)

        if True:
            loss_train = (0.1*self.relu(-(norms-30)) * self.cross_entropy_loss_non_redu(cos_thetas,labels)).mean()
        else:
            loss_train = self.cross_entropy_loss(cos_thetas, labels)
        
        if self.hparams.head == 'cwlmagface':
             ## norms == G_loss at head
            loss_train = loss_train + norms.mean()
        if self.hparams.head == 'magface':
            loss_train = loss_train + 35 * self.head.calc_loss_G(norms)

        
        ## add loss
        #weight_norm = self.head.get_weight_norm()
        
        #max_weight_norm = max(weight_norm)
        #max_weight_norm.require_grad = False
        
        #weight_norm_loss = torch.sqrt(max_weight_norm - weight_norm).mean()
        #loss_train = loss_train + 0.01*weight_norm_loss
       
        ## add loss2 

        #safe_norm = torch.clip(norms,min=0.001,max=100)
        
        #max_norm = max(safe_norm.clone().detach())
        #max_norm.require_grad = False 
        
        #max_norm_loss = torch.sqrt(max_norm - safe_norm).mean()
        #loss_train = loss_train + max_norm_loss
        
        ##########
        
        lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]

        # log
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, logger=True)

        return loss_train

    def training_epoch_end(self, outputs):
        #if self.hparams.head == 'cwcface':
        #    data = {'feature_mb':self.head.feature_mb, 'proxy_mb':self.head.proxy_mb}
        #    with open(f'regist50nms_{self.current_epoch}.pickle','wb') as f:
        #        pickle.dump(data,f)
            

        return None

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        embeddings, norms = self.model(images)

        fliped_images = torch.flip(images, dims=[3])
        flipped_embeddings, flipped_norms = self.model(fliped_images)
        stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
        stacked_norms = torch.stack([norms, flipped_norms], dim=0)
        embeddings, norms = utils.fuse_features_with_norm(stacked_embeddings, stacked_norms)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'output': embeddings,
                'norm': norms,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):

        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            # per dataset evaluation
            embeddings = all_output_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            labels = all_target_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            issame = labels[0::2]
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            num_val_samples = len(embeddings)
            val_logs[f'{dataname}_val_acc'] = acc
            val_logs[f'{dataname}_best_threshold'] = best_threshold
            val_logs[f'{dataname}_num_val_samples'] = num_val_samples

        val_logs['val_acc'] = np.mean([
            val_logs[f'{dataname}_val_acc'] for dataname in dataname_to_idx.keys() if f'{dataname}_val_acc' in val_logs
        ])
        val_logs['epoch'] = self.current_epoch

        for k, v in val_logs.items():
            # self.log(name=k, value=v, rank_zero_only=True)
            self.log(name=k, value=v)

        return None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        test_logs = {}
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            # per dataset evaluation
            embeddings = all_output_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            labels = all_target_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            issame = labels[0::2]
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            num_test_samples = len(embeddings)
            test_logs[f'{dataname}_test_acc'] = acc
            test_logs[f'{dataname}_test_best_threshold'] = best_threshold
            test_logs[f'{dataname}_num_test_samples'] = num_test_samples

        test_logs['test_acc'] = np.mean([
            test_logs[f'{dataname}_test_acc'] for dataname in dataname_to_idx.keys()
            if f'{dataname}_test_acc' in test_logs
        ])
        test_logs['epoch'] = self.current_epoch

        for k, v in test_logs.items():
            # self.log(name=k, value=v, rank_zero_only=True)
            self.log(name=k, value=v)

        return None

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        # if self.trainer.is_global_zero:
        all_output_tensor = torch.cat([out['output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')

        # get rid of duplicate index outputs
        unique_dict = {}
        for _out, _nor, _tar, _dat, _idx in zip(all_output_tensor, all_norm_tensor, all_target_tensor,
                                                all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {'output': _out, 'norm': _nor, 'target': _tar, 'dataname': _dat}
        unique_keys = sorted(unique_dict.keys())
        all_output_tensor = torch.stack([unique_dict[key]['output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor

    def configure_optimizers(self):

        # paras_only_bn, paras_wo_bn = self.separate_bn_paras(self.model)
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)

        optimizer = optim.SGD([{
            'params': paras_wo_bn + [self.head.kernel],
            'weight_decay': 5e-4
        }, {
            'params': paras_only_bn
        }],
                                lr=self.hparams.lr,
                                momentum=self.hparams.momentum)

        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=self.hparams.lr_milestones,
                                             gamma=self.hparams.lr_gamma)

        return [optimizer], [scheduler]

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay
