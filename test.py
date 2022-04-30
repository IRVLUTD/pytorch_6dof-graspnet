from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
from utils.utils import intialize_dataset
import argparse


def run_test(epoch=-1, name=""):
    print('Running Test')
    opt = TestOptions().parse()
    intialize_dataset(opt.dataset)

    opt.serial_batches = True  # no shuffle
    if not opt.model_name:
        opt.name = name
    else:
        opt.name = opt.model_name
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        _, _, ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
