import os
import cv2
import numpy as np
import codecs
import glob

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    # gf_list = glob.glob(os.path.join((data_dir, 'gf')))
    # pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gf_list]
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    # for gt, pred in zip(gt_list, pred_list):
    #     print(gt)
    #     print(pred)
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []

    for pred_path, gt_path in zip(pred_list, gt_list):
        # pred_imgs.append(imread(pred_path))
        pred_img = imread(pred_path)
        pred_img = 255 - pred_img
        pred_imgs.append(pred_img)
        gt_imgs.append(imread(gt_path, thresh=127))


    save_folder = 'save_folder'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, (src_img, tgt_img) in enumerate(zip(pred_imgs, gt_imgs)):
        save_path_src = os.path.join(save_folder, f'src_image_{i}.png')
        save_path_tgt = os.path.join(save_folder, f'tgt_image_{i}.png')
        cv2.imwrite(save_path_src, src_img)
        cv2.imwrite(save_path_tgt, tgt_img)

    print(f'Saved {len(pred_imgs)} images to {save_folder}')
    return pred_imgs, gt_imgs

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f' % v for v in ll])+'\n'
            fout.write(line)

