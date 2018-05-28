import os
import sys
import caffe
import argparse
import numpy as np
import scipy.misc
import cv2
from PIL import Image
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="directory that has the result images")
parser.add_argument("--gt_dir", type=str, default="images/cityscapes_original/", help="directory that has groundtruth images")

parser.add_argument("--output_dir", type=str, default="./result/", help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='caffemodel/', help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--save_output_images", type=int, default=1, help="Whether to save the FCN output images")

args = parser.parse_args()

def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu


def segrun(net, in_):
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)


def fast_hist(a, b, n):
    # print('saving')
    # sio.savemat('/tmp/fcn_debug/xx.mat', {'a':a, 'b':b, 'n':n})

    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
    if len(bc) != n ** 2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)


def preprocess(im):
    """
    Preprocess loaded image (by load_image) for Caffe:
    - cast to float
    - switch channels RGB -> BGR
    - subtract mean
    - transpose to channel x height x width order
    """
    in_ = np.array(im, dtype=np.float32)
    # in_ = in_[:, :, ::-1]
    mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
    in_ -= mean
    in_ = in_.transpose((2, 0, 1))
    return in_

labels = __import__('labels')
id2trainId = {label.id: label.trainId for label in labels.labels}
trainId2color = {label.trainId: label.color for label in labels.labels}

def label_map(label):
    label = np.array(label, dtype=np.float32)
    if sys.version_info[0] < 3:
        for k, v in id2trainId.iteritems():
            label[label == k] = v
    else:
        for k, v in id2trainId.items():
            label[label == k] = v
    return label


def palette(label):
    '''
    Map trainIds to colors as specified in labels.py
    '''
    if label.ndim == 3:
        label= label[0]
    color = np.empty((label.shape[0], label.shape[1], 3))
    if sys.version_info[0] < 3:
        for k, v in trainId2color.iteritems():
	    color[label == k, :] = v
    else:
        for k, v in trainId2color.items():
	    color[label == k, :] = v
    return color

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # read the input and gt txt files

    n_imgs = 500

    classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck',
                'bus', 'train', 'motorcycle', 'bicycle']
    n_cl = len(classes)


    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    net = caffe.Net(args.caffemodel_dir + '/deploy.prototxt',
                    args.caffemodel_dir + '/fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)

    hist_perframe = np.zeros((n_cl, n_cl))
    for i in range(n_imgs):
        if i % 10 == 0:
            print('Evaluating: %d/%d' %(i, n_imgs))

        gt_segmentation_file = os.path.join(args.gt_dir, str(i) + '_label.png')
        #input_im_file = os.path.join(args.result_dir, str(i) + '.jpg')
        input_im_file = os.path.join(args.result_dir, str(i) + '_image.jpg')
        #print input_im_file
        #raw_input('hi')
        label_im = np.array(cv2.imread(gt_segmentation_file))
        label_im = label_im[:,:,0]
        #print np.shape(label_im)

        label_im = label_map(label_im)
        input_im = cv2.imread(input_im_file)
        if np.max(input_im) <= 1.0:
            input_im = input_im * 255.0

        if np.shape(input_im)[:2] != (256,256):
            print 'resizing'
            raw_input('debug stop')
            input_im = cv2.resize(input_im, (256, 256))

        #print np.min(input_im), np.max(input_im)
        #raw_input('fuck')
        input_im = cv2.resize(input_im, (label_im.shape[1], label_im.shape[0]))
        im = preprocess(input_im)
        #input_im = c(input_im_files[i]))[:,:,0:3]
        out = segrun(net, im)
        assert(out.shape == label_im.shape)
        hist_perframe += fast_hist(label_im.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = palette(label_im)
            pred_im = palette(out)
            cv2.imwrite(args.output_dir + '/' + str(i) + '_pred.jpg', pred_im)
            cv2.imwrite(args.output_dir + '/' + str(i) + '_gt.jpg', label_im)
            cv2.imwrite(args.output_dir + '/' + str(i) + '_input.jpg', input_im)

    #pdb.set_trace()
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

main()
