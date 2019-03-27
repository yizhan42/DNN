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
from CNNnd import *

def test_final(model, rp, args, saved_model_name):
    # rp.write(' {:^5s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} \n'.format(
    #     'Mean', 'accuracy', 'accs', 'mcc', 'sens', 'spec', 'f1'))
    targets, predicts, es, accs = [], [], np.zeros(7), 0
    for i in range(args.start, args.end):
        
        # data_path = '{}/{}_{}.csv'.format(
        #     args.test_data_folder, args.prefix_filename, i)
        data_path = './data/multihot_data/test/test_joint_without_id.csv'
        model_path = '{}/{}_{}/{}.pth.tar'.format(
            args.model_path, args.prefix_groupname, i, saved_model_name)


        print("=> loading parameters from '{}'".format(model_path))
        assert os.path.isfile(model_path), "=> no checkpoint found at '{}'".format(model_path)
        if args.cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(
                model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        # using GPU
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
       
        print('Loading Testing data from {}'.format(data_path))
        test_dataset = readTestData(
            # label_data_path='{}{}'.format(args.test_data_folder, args.prefix_filename),
            data_path,
            # index = i,
            # total=args.groups,
        )
        test_data = ProteinDataset(
            pd_dataFrame=test_dataset,
        )
        test_x = [test_data[i][0] for i in range(len(test_data))]
        # print(test_x)
        # test_x = Variable(torch.unsqueeze(torch.Tensor(test_x)))
        test_x = Variable(torch.Tensor(test_x)).reshape(len(test_x),1,4221)
        test_y = torch.from_numpy(test_data.labels).long()

        if args.cuda:
            test_x, test_y = test_x.cuda(), test_y.cuda()
        
        test_output = model(test_x[:])
        score = test_output.cpu().data.squeeze().numpy()
        # score = torch.max(test_output, 1)[0].data.squeeze()
        preds = torch.max(test_output,1)[1].data.squeeze()
        #print(preds)
        # print(test_data.labels,type(test_data.labels))
        labels = (test_data.labels,[1-label for label in test_data.labels])

        # accuracy, sens, spec, ppv, npv, f1, mcc, acc[neg, pos]
        evaluations, accuracy = evaluate(test_y, preds)

        predicts.append(preds)
        targets.append(test_y)
        es += evaluations
        accs+= sum(accuracy)/2
    
    es /= args.end - args.start
    accs /= args.end - args.start
    
    rp.write(' {:^5s} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} | {:10f} \n'.format(
        'Mean', es[0], accs, es[6], es[1], es[2], es[5]))

    drawMeanRoc(
        targets, predicts, pos_label=1, is_show=False,
        save_file='{}/{}_roc.png'.format(args.model_path, saved_model_name))


def runAndDraw(model, args):  
    with open('{}/analysis.csv'.format(args.model_path), 'w') as rp:
        rp.write(' {:^5s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} | {:10s} \n'.format(
        'Mean', 'accuracy', 'accs', 'mcc', 'sens', 'spec', 'f1'))
        test_final(model(), rp, args, saved_model_name='best_accuracy')
        test_final(model(), rp, args, saved_model_name='best_loss')
        
    result_log_files = []
    for i in range(args.start, args.end):
        result_log_files.append(
            '{}/{}_{}/result.csv'.format(args.model_path, args.prefix_groupname, i))
    drawLossFigureWhenCrossValidatingInMultFiles(
        result_log_files, save_path=args.model_path, is_print=False, is_save=True)
    

if __name__ == "__main__":
    
    model = CNN_multihot
    args = parser.parse_args()
    runAndDraw(model,args)
