import argparse
import os

import torch
from tqdm import tqdm

from src import EBVGCDataset, EfficientNet


def prepare_custom_dataset(data_dir, annotation=None):
    dataset = []
    for path, dir, files in os.walk(data_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            file_path = os.path.join(path, filename)
            label = 0 if annotation is None else annotation[file_path]
            dataset.append((file_path, label))
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EBV-TRACER Classification Inference')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--checkpoint', default="./checkpoints/net_5000.pth", type=str, help='path to latest checkpoint')
    parser.add_argument('--workers', default=16, type=int, help='number of data_function loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--results', default="./results.csv", type=str, help='path to results')
    args = parser.parse_args()

    model = EfficientNet.from_name("class", "efficientnet-b1", num_classes=3)
    model.load_state_dict(torch.load(args.checkpoint))
    if torch.cuda.is_available():
        model = model.cuda()

    # Dataset
    test_set = prepare_custom_dataset(args.data, annotation=None)
    test_dataset = EBVGCDataset(test_set, input_size=args.input_size, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              shuffle=False)

    wf = open(args.result, 'w')
    wf.write('file,pred\n')

    model.eval()
    tqdm_desc = 'Evaluation {}'.format(args.data_name)
    with torch.no_grad():
        for i, (input_paths, inputs, _) in tqdm(enumerate(test_loader), desc=tqdm_desc):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            output = model(inputs)
            preds = torch.argmax(output, dim=1)

            for input_path, p in zip(input_paths, preds):
                wf.write('{},{}\n'.format(input_path, p))
    wf.close()
