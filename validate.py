
import torch
# from FeatureSelectionData import *

from test_final import *
from CNNnd import *
from CNNst import *
from torch.autograd import Variable
from load_data import *

# from train import *


def validate(model, validation_data, args):
    X_w = Variable(torch.from_numpy(validation_data.wide)).float()
    X_d = Variable(torch.from_numpy(validation_data.deep))
    target = Variable(torch.from_numpy(validation_data.labels)).float()
    if args.cuda:
        X_w, X_d, target = X_w.cuda(), X_d.cuda(), target.cuda()

    net = model.eval()
    pred_score = net(X_w,X_d).cpu()

    val_loss = model.criterion(pred_score, target,reduction='sum')

    if model.method == "regression":
        pred_y = pred_score.squeeze(1).data
    if model.method == "logistic":
        pred_y = (pred_score > 0.5).squeeze(1).float()
    if model.method == "multiclass":
        _, pred_y = torch.max(pred_score, 1)
    # print(target, pred_score)
    #print(pred_y)
    #print(target)
    val_accuracy =  float((pred_y == target).sum().item())/target.size(0)
    # validation_loss = val_loss.data[0] / validation_y.size(0)
    validation_loss = val_loss.data / target.size(0)
    model.train()
    return val_accuracy,validation_loss

if __name__ == "__main__":
    # model = CNN_multihot()
    args = parser.parse_args()
    # i = 0
    # train_dataset, validation_dataset = readTrainingData(
    #     label_data_path='{}{}'.format(args.train_data_folder, args.prefix_filename),
    #     index=i,
    #     total=args.groups,
    #     # standard_length=args.length,
    # )
    model = WideDeep
    validate(model,validation_dataset, args)
    print("Done")
