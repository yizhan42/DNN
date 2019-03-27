
import torch
from FeatureSelectionData import *

from test_final import *
from torch.autograd import Variable


# from train import *


def validate(model, validation, args):
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_loss = checkpoint.get('best_loss', None)
        best_acc = checkpoint.get('best_acc', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
    else:
        best_loss = None
        best_acc = None

    global sum_val
    torch.manual_seed(1)    # reproducible

    # put model in GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    validation_data = ProteinDataSet(
        pd_dataFrame = validation,
        transform=None
    )

    validation_x = [validation_data[i][0] for i in range(len(validation_data))]
    validation_x = Variable(torch.Tensor(validation_x))
    validation_y = torch.from_numpy(validation_data.labels)
    validation_y = validation_y.float()
    validation_y = Variable(validation_y, requires_grad=False)

    criterion = torch.nn.MSELoss(size_average=False)

    if args.cuda:
        validation_x, validation_y = validation_x.cuda(), validation_y.cuda()

    pred_score = model(validation_x)

    pred_y = torch.max(pred_score, 1)[0]

    val_loss = criterion(pred_y, validation_y)

    val_accuracy = torch.sum(torch.max(pred_score, 1)[1].data.squeeze() == validation_y.data.long()).numpy() / float(
        validation_y.size(0))

    # validation_loss = val_loss.data[0] / validation_y.size(0)
    validation_loss = val_loss.data / validation_y.size(0)

    return val_accuracy,validation_loss

if __name__ == "__main__":
    model = TwoLayerNet(D_in, H, I, J, K, D_out)
    args = parser.parse_args()
    i = 0
    train_dataset, validation_dataset = readTrainingData(
        label_data_path='{}{}'.format(args.train_data_folder, args.prefix_filename),
        index=i,
        total=args.groups,
        # standard_length=args.length,
    )
    validate(model,validation_dataset, args)
    print("Done")