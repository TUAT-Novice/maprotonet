import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-d', '--data-path', type=str, required=True, help="path to the data files")
    parser.add_argument('--n-workers', type=int, default=8, help="number of workers in data loader")
    parser.add_argument('--n-threads', type=int, default=4, help="number of CPU threads")
    parser.add_argument('--augmented', type=int, choices={0, 1}, default=1,
                        help="whether to perform data augmentation during training")
    parser.add_argument('--aug-seq', type=str, default=None, help="data augmentation sequence")

    # training
    parser.add_argument('-n', '--epoch', type=int, default=100, help="maximum number of epochs to train on")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help="batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="peak value for learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay parameter")
    parser.add_argument('--bc-opt', type=str, choices={'Off', 'BCE', 'B2CE', 'CBCE', 'FL', 'BFL', 'B2FL', 'CBFL'},
                        default='BFL', help="balanced classification option")
    parser.add_argument('--op-opt', type=str, choices={'Adam', 'AdamW'}, default='AdamW', help="optimizer option")
    parser.add_argument('--lr-opt', type=str, choices={'Off', 'StepLR', 'CosALR', 'WUStepLR', 'WUCosALR'},
                        default='WUCosALR', help="learning rate scheduler option")
    parser.add_argument('--lr-n', type=int, default=0, help="learning rate scheduler number")
    parser.add_argument('--wu-n', type=int, default=0, help="number of warm-up epochs")
    parser.add_argument('--accumulate-step', type=int, default=1, help="gradient accumulate steps")
    parser.add_argument('--use-amp', type=int, choices={0, 1}, default=1,
                        help="whether to use automatic mixed precision")
    parser.add_argument('--use-da', type=int, choices={0, 1}, default=0,
                        help="whether to use deterministic algorithms")

    # model
    # base for all models
    parser.add_argument('-m', '--model-name', type=str,
                        choices={'maprotonet', 'mprotonet', 'xprotonet', 'protopnet', 'cnn'},
                        required=True, help="name of the model")
    parser.add_argument('--backbone', type=str, default='resnet152_quad',
                        choices={'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnet18_quad', 'resnet34_quad', 'resnet50_quad', 'resnet101_quad', 'resnet152_quad'},
                        help='backbone model for feature extraction')
    parser.add_argument('--n_res_block', type=int, choices={0, 1, 2, 3, 4}, default=1,
                        help='number of residual block for backbone model')
    # only for protopnet
    parser.add_argument('--topk_p', type=int, default=1, help='topk most region similar scores for features')
    # for protopnet, xprotonet, mprotoNet, maprotonet
    parser.add_argument('--prototype_shape', type=str, default='(30, 128, 1, 1, 1)',
                        help='dimension of the prototype vectors')
    parser.add_argument('--f_dist', type=str, choices={'cos', 'l2'}, default='cos',
                        help='metric to evaluate distance between prototype vectors and features')
    parser.add_argument('--coefs', type=str,
                        default='{"cls": 1, "clst": 0.8, "sep": -0.08, "L1": 0.01, "map": 0.5, "OC": 0.05}',
                        help='weights for every loss item')
    # only for maprotonet
    parser.add_argument('--n-scales', type=int, default=1, help='number of scales for p_map')
    parser.add_argument('-mm', '--mmloss', type=int, choices={0, 1}, default=1,
                        help="whether to use multi-scale mapping loss")
    parser.add_argument('--scale-mode', type=str, choices={'a', 'b', 'c', 'd'}, default='c',
                        help='architecture for multi-scale features')

    # evaluate
    parser.add_argument('--attr', type=str,
                        help='attribution method, please provide the combination of {P, D, G, U, O}, representing'
                             'ProtoNets, Deconvolution, GradCAM, Guided GradCAM and Occlusion, respectively.')

    # others
    parser.add_argument('--load-model', type=str, default=None, help="whether to load the model")
    parser.add_argument('--save-model', type=int, choices={0, 1}, default=0, help="whether to save the best model")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")

    return parser.parse_args()
