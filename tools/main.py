'''
Created on Sep 3, 2017

@author: georgeretsi

do not support multi-gpu yet. needs thread manipulation
- works on GW + IAM
- new way to load dataset
- augmentation with dataloader
- Hardcoded selections (augmentation - default:YES, load pretrained model with hardcoded name...
- do not normalize with respect to iter size (or batch size) for speed
- add fixed size selection (not hardcoded)
- save and load hardcoded name 'PHOCNet.pt'
'''
import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import tqdm
import sys

sys.path.insert(0, '/home/suman/PycharmProjects/DeformHandWriting')
# sys.path.insert(0, '/home/suman/DeformHandWriting')
# sys.path.insert(0, '/home/suman/DeformHandWriting')
import copy
import torch.multiprocessing
from dataset import collate_iam as cv
torch.multiprocessing.set_sharing_strategy('file_system')
from dataset.iam_alt import IAMDataset
from dataset.synth_text import SynthDataSet
from dataset.gw_alt import GWDataset
sys.path.insert(0, '/home/suman/PycharmProjects/sphoc')
#from dataset.gw_alt import GWDataset

# from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
# from cnn_ws.losses.cosine_loss import CosineLoss

# from cnn_ws.models.myphocnet import PHOCNet
# from cnn_ws.models.encdec import Autoencoder
# from cnn_ws.evaluation.retrieval import map_from_feature_matrix, map_from_query_test_feature_matrices
# from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler
from src.model import Autoencoder
from torch.nn.utils.rnn import  pack_padded_sequence
from util.cuda import to_var
# from src.dict_net import *

# from cnn_ws.utils.save_load import my_torch_save, my_torch_load


def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]


def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')

    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser,
                        default='30000:1e-3,60000:1e-4,150000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=16,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--test_batch_size', '-tbs', action='store', type=int, default=8,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    # parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default=0,
    #                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # - experiment arguments
    parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN. Default: 26')
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,3,4,5,6,7,8,9,10',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc', 'spoc', 'dctow', 'phoc-ppmi', 'phoc-pruned'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc, spoc, phoc-ppmi, phoc-pruned. Default: phoc')
    parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default='70,150',
                        help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
    parser.add_argument('--dataset', '-ds', required=False, choices=['gw', 'iam', 'synth'], default='gw',
                        help='The dataset to be trained on')
    args = parser.parse_args()

    # sanity checks
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # dict_weights = '../models/dictnet_vgg_conv.caffemodel'
    # dict_proto = '../models/dictnet_vgg_deploy_conv.prototxt'
    # dictnet = dict_net(protofile=dict_proto, weightfile=dict_weights)
    # dictnet = torch.load('dictnet.pkl')
    # net = torch.load('/home/suman/Desktop/sphoc-pytorch/tools/SPHOC_len_char.pt')

    # prepare datset loader
    # TODO: add augmentation
    logger.info('Loading dataset %s...', args.dataset)
    if args.dataset == 'gw':
        train_set = GWDataset(gw_root_dir='/home/suman/PycharmProjects/sphoc/data/gw',
                              cv_split_method='almazan',
                              cv_split_idx=1,
                              image_extension='.tif',
                              embedding=args.embedding_type,
                              phoc_unigram_levels=args.phoc_unigram_levels,
                              fixed_image_size=args.fixed_image_size,
                              min_image_width_height=args.min_image_width_height)
        test_set = copy.copy(train_set)
        train_set.mainLoader(partition='train')
        test_set.mainLoader(partition='test', transforms=None)

        # augmentation using data sampler
        n_train_images = 500000
        augmentation = True

    if args.dataset == 'iam':
        train_set = IAMDataset(gw_root_dir='../../DeformHandWriting/data/IAM',
                               image_extension='.png',
                               embedding=args.embedding_type,
                               phoc_unigram_levels=args.phoc_unigram_levels,
                               fixed_image_size=args.fixed_image_size,
                               min_image_width_height=args.min_image_width_height)
        test_set = copy.copy(train_set)
        train_set.mainLoader(partition='train')
        test_set.mainLoader(partition='test', transforms=None)

        # augmentation using data sampler
        n_train_images = 500000
        augmentation = True
    if args.dataset == 'synth':
        train_set = SynthDataSet(root_dir='/media/suman/B0A68CFCA68CC476/mnt/ramdisk/max/90kDICT32px/')
        test_set = SynthDataSet(root_dir='/media/suman/B0A68CFCA68CC476/mnt/ramdisk/max/90kDICT32px/')
        augmentation = False

    criterion = nn.CrossEntropyLoss()
    criterionMSE = nn.MSELoss()

    if augmentation:
        train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images),
                                  batch_size=args.batch_size,
                                  num_workers=1, collate_fn=cv.pad_collate)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=1,collate_fn=cv.pad_collate)

    # train_loader_iter = iter(train_loader)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=8,
                             collate_fn=cv.pad_collate)
    # load CNN
    logger.info('Preparing PHOCNet...')

    # cnn = PHOCNet(n_out=train_set[0][1].shape[0],
    #              input_channels=1,
    #              gpp_type='gpp',
    #              pooling_levels=([1], [5]))
    model = Autoencoder()

    max_epochs = 128
    if args.gpu_id is not None:
        if len(args.gpu_id) > 1:
            model = nn.DataParallel(model, device_ids=args.gpu_id)
            model.cuda()
            # net.cuda()
        else:
            model.cuda(args.gpu_id[0])
            # net.cuda(args.gpu_id[0])
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # evaluate_cnn_batch(cnn=model,
    #                    dataset_loader=test_loader,
    #                    args=args, loss_fn=criterion,
    #                    net=None)

    for epoch in range(max_epochs):
        for batch_idx, data_batch in enumerate(train_loader):
            optim.zero_grad()
            im, embedding, lengths = data_batch
            im = to_var(im)
            #print(im.shape)
            #feats = net.encoder(im)
            #print(feats.shape)
            #feats = to_var(feats)
            embedding = to_var(embedding)
            lengths = to_var(lengths)
            #print(embedding)
            #print(lengths)
            out, out_captions, lengths_out, alphas,out_image = model(im, embedding, lengths)
            #print(lengths_out)
            predicts = pack_padded_sequence(out, [l for l in lengths_out], batch_first=True)[0]
            targets = pack_padded_sequence(out_captions[:, 1:], [l for l in lengths_out], batch_first=True)[0]
            loss_val = criterion(predicts, targets)
            x_recons = torch.sigmoid(out_image)
            Lx = criterionMSE(x_recons,im.reshape(im.shape[0],10500))
            loss = loss_val+Lx

            #print(predicts)
            #print(loss_val.item())
            loss.backward()
            optim.step()
            if (batch_idx + 1) % args.display == 0:
                logger.info('Iteration %d: %f, %f',  batch_idx + 1, loss_val.item(),Lx.item())
            if (batch_idx) % args.test_interval == 0:
                evaluate_cnn_batch(cnn=model,
                                   dataset_loader=test_loader,
                                   args=args, loss_fn=criterion)


def evaluate_cnn_batch(cnn, dataset_loader, args, loss_fn):
    logger = logging.getLogger('PHOCNet-Experiment::test')
    # set the CNN in eval mode
    # fh = logging.FileHandler('log1.txt')
    # logger.addHandler(fh)
    cnn.eval()
    logger.info('Computing net output:')
    # qry_ids = np.zeros((len(dataset_loader), args.test_batch_size), dtype=np.int32)
    # class_ids = np.zeros((len(dataset_loader), args.test_batch_size), dtype=np.int32)
    # embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = []
    # np.zeros((len(dataset_loader), args.test_batch_size, embedding_size), dtype=np.float32)
    outputs = []
    # np.zeros((len(dataset_loader), args.test_batch_size, embedding_size), dtype=np.float32)
    lengths_all = np.zeros((len(dataset_loader), args.test_batch_size), dtype=np.int)
    loss = 0.0
    nb = 0
    acc_all =[]
    predw_all=[]
    gtw_all =[]
    for lc, test_batch in enumerate(tqdm.tqdm(dataset_loader)):
        # if sample_idx > 10000:
        #     break
        word_img, embedding, lengths = test_batch
        im = to_var(word_img)
        # feats = net.encoder(im)
        # print(feats.shape)
        # print(feats['conv4'].shape)
        # feats = to_var(feats['conv4'])
        embedding = to_var(embedding)
        lengths = to_var(lengths)
        out = cnn.sampler(im)
        # print(out.shape)
        acc, predw, gtw = get_accuracy(out, embedding, lengths)
        acc_all.append(acc)
        predw_all.extend(predw)
        gtw_all.extend(gtw)
        # print(out)
        # predicts = pack_padded_sequence(out, [l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
        # targets =  pack_padded_sequence(embedding[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
        # loss_val = loss_fn(predicts, targets)

        # loss = loss + (loss_val * out.shape[0])
        # print(out.shape)
        # output = torch.softmax(out)
        # outputs.append(out.data.cpu().numpy())
        # embeddings.append(embedding.data.cpu().numpy())
        # lengths_all[lc] = lengths.data.cpu().numpy()

    accuracy = np.mean(np.array(acc))
    f=open('output.txt', 'w')
    for cnt, gt in enumerate(gtw_all):
        f.write(predw_all[cnt] + ' ' + gt+'\n')
    f.close()
    logger.info('Accuracy: %3.2f %f', accuracy, loss)
    cnn.train()
    #     # print(loss)
    #     # nb = nb + 1
    #     # outputs[lc] = output.data.cpu().numpy()
    #     # embeddings[lc] = embedding.data.cpu().numpy()
    #     # # print(class_id.shape)
    #     # class_ids[lc] = class_id.numpy().flatten()
    #     # # print is_query.shape
    #     # qry_ids[lc] = is_query.byte().numpy().flatten()
    #     # print qry_ids
    #     # if is_query[0] == 1:
    #     # qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]
    #
    # '''
    # # find queries
    #
    # unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    # qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]
    #
    # # remove stopwords if needed
    #
    # qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    # '''
    # qry_ids = qry_ids.flatten()
    # qry_ids = np.where(qry_ids == 1)[0]
    # # print(qry_ids)
    # outputs = outputs.reshape((-1, embedding_size))
    # class_ids = class_ids.flatten()
    # qry_outputs = outputs[qry_ids]
    # qry_class_ids = class_ids[qry_ids]
    # # print(outputs.shape)
    # # print(qry_outputs.shape)
    #
    # # run word spotting
    # logger.info('Computing mAPs...')
    #
    # _, ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
    #                                                         test_features=outputs,
    #                                                         query_labels=qry_class_ids,
    #                                                         test_labels=class_ids,
    #                                                         metric='cosine',
    #                                                         drop_first=True)
    # # print(ave_precs_qbe)
    # mAP = np.mean(ave_precs_qbe[ave_precs_qbe > 0]) * 100
    # loss = loss / np.float(nb)
    #
    # # logger.info('mAP: %3.2f %f', mAP, loss)
    # # clean up -> set CNN in train mode again
    # cnn.train()
    # return loss, mAP


def get_accuracy(prediction, targets, lengths):
    # _, pred_ids = torch.max(prediction, dim=1)
    corr =0
    unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9')+1))]
    indices = range(1, 37)
    unigram_dict = dict(zip(indices,unigrams))
    unigram_dict[0] = '@'
    unigram_dict[37] = '-'
    unigram_dict[38] = '/'
    #print(unigram_dict)

    prediction_words =[]
    target_words=[]
    if len(prediction.shape)==1:
        prediction = prediction.reshape(1,-1)
    for ind in range(prediction.shape[0]):
        pred = prediction[ind].cpu().numpy()
        predw=[]
        target = targets[ind][:lengths[ind]-1].cpu().numpy()
        targetw =[]
        # print(pred)
        # print(target)
        for v in pred:
            if v ==37:
                break
            predw.append(unigram_dict[v])
        for v in target:
            targetw.append(unigram_dict[v])
        # predw.append(chr(v))
        targetw = ''.join(targetw)
        predw = ''.join(predw)
        #print(targetw)
        #print(predw)

        # corrects = (prediction[ind][:lengths[ind]].data == targets[ind][:lengths[ind]].data)
        if predw == targetw[1:]:
            corr = corr+1
        prediction_words.append(predw)
        target_words.append(targetw)

    # print(corr)
    acc = corr/np.float(prediction.shape[0])
    # torch.mean(corrects.float())
    return acc, prediction_words, target_words

def evaluate_cnn(cnn, dataset_loader, args):
    logger = logging.getLogger('PHOCNet-Experiment::test')
    # set the CNN in eval mode
    cnn.eval()
    logger.info('Computing net output:')
    qry_ids = []  # np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm.tqdm(dataset_loader)):
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            # word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''
        output = torch.sigmoid(cnn(word_img))
        # output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0, 0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  # [sample_idx] = is_query[0]

    '''
    # find queries

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed

    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                         test_features=outputs,
                                                         query_labels=qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0]) * 100)

    # clean up -> set CNN in train mode again
    cnn.train()


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
