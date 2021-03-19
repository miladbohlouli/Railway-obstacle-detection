import argparse

def get_args():
    parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')

    # model architecture & checkpoint
    parser.add_argument('--model', metavar='[fastercnn_resnet50_fpn]',
                        default='fastercnn_resnet50_fpn', type=str, help='model')
    parser.add_argument('--save_path', metavar='path/to/save_results', default='saved_results',
                        type=str, help='path to results saved')
    parser.add_argument('--norm', metavar='[batch, instance, evo, group]', default='batch',
                        type=str, help='replace batch norm with other norm')

    # data loading
    parser.add_argument('--source_video', metavar='path/to/source/video', default='./sample_videos/1.mp4',
                        type=str, help='path to dataset (ends with /minicity)')
    parser.add_argument('--pin_memory', metavar='[True,False]', default=True,
                        type=bool, help='pin memory on GPU')
    parser.add_argument('--num_workers', metavar='8', default=0, type=int,
                        help='number of dataloader workers')

    # data augmentation hyper-parameters
    parser.add_argument('--image_size', default=[50, 50], nargs='+', type=int, help='image size during training')
    parser.add_argument('--dataset_mean', metavar='[0.485, 0.456, 0.406]',
                        default=[0.485, 0.456, 0.406], type=list,
                        help='mean for normalization')
    parser.add_argument('--dataset_std', metavar='[0.229, 0.224, 0.225]',
                        default=[0.229, 0.224, 0.225], type=list,
                        help='std for normalization')
    parser.add_argument('--threshold', type=float, default=0.9, help="The threshold the boxes will be chosen with")


    # appearance parameeters
    parser.add_argument('--text_size', default=3, help='Text size used for writing on the images')
    parser.add_argument('--text_thickness', default=3, help='Thickness of the written text')
    parser.add_argument('--rectangle_thickness', default=3, help='Thickness of the rectangle border')
    parser.add_argument('--color', default=(0, 255, 0), help='color for drawing on the image')

    args = parser.parse_args()
    return args