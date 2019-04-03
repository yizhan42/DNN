import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from analysis import *
# from train import *
# from dnn_main import *
from load_data import *
from scipy import interp
from settings import *
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from analysis import evaluate
# from CNNnd import *
# from CNNst import *
from train import *
from wide_deep.torch_model import *
from wide_deep.data_utils import prepare_data


DF = pd.read_csv('data/wdl_data/test/test_joint_without_id.csv', header=None)
wide_cols = [x for x in range(1000,4000)]  
crossed_cols = ()
embeddings_cols = [(3,4),(5,7),(7,8),(2,3)]
continuous_cols = [8,9]  
target = 0  
method = 'logistic'
hidden_layers = [100,50]
dropout = [0.5,0.2]

wd_dataset = prepare_data(
    DF, wide_cols,
    crossed_cols,
    embeddings_cols,
    continuous_cols,
    target,
    scale=True)
test_data = wd_dataset['dataset']

def test_final(model, rp, args, saved_model_name, test_data):
    model.compile(method=method)
    
    targets, predicts, es, accs = [], [], np.zeros(7), 0
    # using GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    for i in range(args.start, args.end):
        data_path = '{}/test/test_joint_without_id.csv'.format(args.data_folder)
        model_path = '{}/{}_{}/{}.pth.tar'.format(
            args.save_folder, args.prefix_groupname, i, saved_model_name)
        print("=> loading parameters from '{}'".format(model_path))
        assert os.path.isfile(model_path), "=> no checkpoint found at '{}'".format(model_path)
        if args.cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(
                model_path, map_location=lambda storage, loc: storage)
               
        model.load_state_dict(checkpoint['state_dict'])
       
        print('Loading Testing data from {}'.format(data_path))
  
        X_w = Variable(torch.from_numpy(test_data.wide)).float()
        X_d = Variable(torch.from_numpy(test_data.deep))
        target = Variable(torch.from_numpy(test_data.labels)).float()
        if args.cuda:
            X_w, X_d, target = X_w.cuda(), X_d.cuda(), target.cuda()

        net = model.eval()
        test_output = net(X_w,X_d)
        score = test_output.cpu()
        # score = test_output.cpu().data.squeeze().numpy()      
        preds = torch.max(test_output,1)[1].data.squeeze()       
        # labels = (test_data.labels,[1-label for label in test_data.labels])
        
        if model.method == "regression":
            pred_y = score.squeeze(1).data.numpy()
        if model.method == "logistic":
            pred_y = (score > 0.5).squeeze(1).data.numpy()
        if model.method == "multiclass":
            _, pred_y = torch.max(score, 1)
       
      # accuracy, sens, spec, ppv, npv, f1, mcc, acc[neg, pos]
        evaluations, accuracy = evaluate(target, preds)

        rp.write(' {:^5d} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} \n'.format(
            i, evaluations[0], sum(accuracy)/2, evaluations[6], evaluations[1], evaluations[2], evaluations[5]))

        predicts.append(preds.cpu())
        targets.append(target.cpu())
        es += evaluations
        accs+= sum(accuracy)/2
    
    es /= args.end - args.start
    accs /= args.end - args.start
    
    rp.write(' {:^5s} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} \n'.format(
        'Mean', es[0], accs, es[6], es[1], es[2], es[5]))

    drawMeanRoc(
        targets, predicts, pos_label=1, is_show=False,
        save_file='{}/{}_roc.png'.format(args.save_folder, saved_model_name))

    drawMeanPR(
        targets, predicts, pos_label=1, is_show=False,
        save_file='{}/{}_pr_fragments.png'.format(args.model_path, saved_model_name))


def runAndDraw(model, args):  
    with open('{}/analysis.csv'.format(args.save_folder), 'w') as rp:
        rp.write(' {:^5s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} \n'.format(
        'Group', 'accuracy', 'mean accuracy', 'mcc', 'sens', 'spec', 'f1'))
        # test_data = wd_dataset['dataset']
        wide_dim = wd_dataset['dataset'].wide.shape[1]
        n_unique = len(np.unique(wd_dataset['dataset'].labels))
        if (method=="regression") or (method=="logistic"):
            n_class = 1
        else:
            n_class = n_unique
        deep_column_idx = wd_dataset['deep_column_idx']
        embeddings_input= wd_dataset['embeddings_input']
        encoding_dict   = wd_dataset['encoding_dict']
        
        test_final(model(wide_dim,
            embeddings_input,
            continuous_cols,
            deep_column_idx,
            hidden_layers,
            dropout,
            encoding_dict,
            n_class), rp, args, 'best_accuracy',test_data)
        test_final(model(wide_dim,
            embeddings_input,
            continuous_cols,
            deep_column_idx,
            hidden_layers,
            dropout,
            encoding_dict,
            n_class), rp, args, 'best_loss',test_data)
        
    result_log_files = []
    for i in range(args.start, args.end):
        result_log_files.append(
            '{}/{}_{}/result.csv'.format(args.save_folder, args.prefix_groupname, i))
    drawLossFigureWhenCrossValidatingInMultFiles(
        result_log_files, save_path=args.save_folder, is_print=False, is_save=True)
    

if __name__ == "__main__":
    
    # model = CNN_knnscore
    args = parser.parse_args()
    model = WideDeep
    runAndDraw(model,args)
