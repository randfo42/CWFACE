import torch
import numpy as np
from tqdm import tqdm
import data_utils
import argparse
import pandas as pd
import survface_helper
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import net


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet. please refer to https://github.com/deepinsight/insightface/blob/5d3be6da49275602101ad122601b761e36a66a01/recognition/_evaluation_/ijb/ijb_11.py#L296')
        # note that they do not use normalization afterward.
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


def infer(model, dataloader, use_flip_test, fusion_method):
    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):

            feature = model(images.to("cuda:0"))
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                norm = None

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda:0"))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_norm = None

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    norms = np.concatenate(norms, axis=0)
    return features, norms

def load_pretrained_model(model_name='ir50'):
    # load model and pretrained statedict
    ckpt_path = adaface_models[model_name][0]
    arch = adaface_models[model_name][1]

    model = net.build_model(arch)
    statedict = torch.load(ckpt_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--data_root', default='/home/taes/face_data')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--model_name', type=str, default='ir101_webface4m')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add', choices=('average', 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score'))
    args = parser.parse_args()

    # load model
    adaface_models = {
        #'ir50': ["../pretrained/adaface_ir50_ms1mv2.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/cwface_ir50_ms1mv2.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_ada_20_01023.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adalimit25_0602_ep32.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adaindex_0605_ep37.ckpt", 'ir_50'],
        'ir50': ["../pretrained/ir50_casia_adaface_07131_38ep.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adalimit_07132_34ep.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adalimitstd0_07150_39ep.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adawindex_07171_ep34.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_adawindex_bm10std10mse20_07181_ep37.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_ada_cw_weight_08115_ep34.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_ada_kl_fnweight_08160_ep38.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_ada_kl_fnweight_08170_ep33.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_fratface_08302_ep31.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_fratface_09302_ep31.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_qualface_qd_09-14_0/epoch=34-step=33564.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_10-12___12/epoch=34-step=33564.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight2_10-13_0/epoch=38-step=37400.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight3_10-14_0/epoch=34-step=33564.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight4_10-14_0/epoch=33-step=32605.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight5_10-15_0/epoch=37-step=36441.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight6_10-16_0/epoch=31-step=30687.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight3_filter0502_10-16_0/epoch=28-step=27810.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight1_proj_10-17_8/epoch=33-step=32605.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight2_proj_0502_10-17_0/epoch=37-step=36441.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight2_proj_0502_10-18_0/epoch=38-step=37400.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight0_F_proj_0500_10-18_0/last.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight1_proj_layer_0500_10-19_________18/epoch=35-step=34523.ckpt", 'ir_50'],
        
        #'ir50': ["../experiments/ir50_ufo_weight1_proj_layer_wo_rel_0500_10-20_0/epoch=29-step=28769.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight1_proj_layer_wo_rel_0502_10-21_0/epoch=28-step=27810.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight1_proj_layer2_rel_0502_10-21_0/epoch=33-step=32605.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight1_time_proj_layer2_rel_0502_10-22_2/epoch=39-step=38359.ckpt", 'ir_50'],
        #'ir50': ["../experiments/ir50_ufo_weight6_time_cos_0502_10-23_8/epoch=32-step=31646.ckpt", 'ir_50'],
        # 'ir50': ["../experiments/ir50_ufo_weight7_time_cos_0502_10-24_0/epoch=38-step=37400.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_cw_25_012018.ckpt", 'ir_50'],
        #'ir50': ["../pretrained/ir50_casia_cwl_25_01289.ckpt", 'ir_50'],
        #'ir101_ms1mv2': ["../pretrained/adaface_ir101_ms1mv2.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_ada_29_01280.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cwl_23_02080.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cwl_clip_24_02170.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cw_cwl_clip_25_03080.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cwc_24_03181.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cwc_ncomp_25_03241.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_cwc_ncomp_25_03241.ckpt", 'ir_101'],
        #'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_ada_limit25_34_06122.ckpt", 'ir_101'],
        'ir101_ms1mv2': ["../pretrained/ir101_ms1mv2_ada_limit22_28_06191.ckpt", 'ir_101'],
        #'ir101_ms1mv3': ["../pretrained/adaface_ir101_ms1mv3.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["../pretrained/ir101_ms1mv3_adalimit25_ep29_06271.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["../experiments/ir101_ada_ms1mv3_fratface_10-07_0/epoch=22-step=465358.ckpt", 'ir_101'],
        'ir101_webface4m': ["../pretrained/adaface_ir101_webface4m.ckpt", 'ir_101'],
        'ir101_webface12m': ["../pretrained/adaface_ir101_webface12m.ckpt", 'ir_101'],
    }
    assert args.model_name in adaface_models
    # load model
    model = load_pretrained_model(args.model_name)
    model.to('cuda:{}'.format(args.gpu))

    survface_test = survface_helper.TinyFaceTest(tinyface_root=args.data_root,
                                                 alignment_dir_name='surv_align')

    # set save root
    gpu_id = args.gpu
    save_path = os.path.join('./survface_result', args.model_name, "fusion_{}".format(args.fusion_method))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path: {}'.format(save_path))

    img_paths = survface_test.image_paths
    print('total images : {}'.format(len(img_paths)))
    dataloader = data_utils.prepare_dataloader(img_paths,  args.batch_size, num_workers=0)
    features, norms = infer(model, dataloader, use_flip_test=args.use_flip_test, fusion_method=args.fusion_method)
    results = survface_test.test_identification(features, ranks=[1,5,20])
    print(results)
    pd.DataFrame({'rank':[1,5,20], 'values':results}).to_csv(os.path.join(save_path, 'result.csv'))
