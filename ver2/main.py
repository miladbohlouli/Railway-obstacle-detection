from utils import get_dataloader, get_model, get_device, get_transforms, save_video
from arguments import get_args
from dataset import dataset
from evaluation import evaluate

def main():
    args = get_args()
    model = get_model(args)
    transforms = get_transforms(args)
    ds = dataset(transforms, args)
    data_loader = get_dataloader(ds, args)
    device = get_device()

    # Running the evaluation on the inout video
    print(f"____Running the model on {device}___\n")
    edited_frames = evaluate(model, data_loader, device, args)

    print(f"____saving the video in {args.save}___\n")
    save_video(edited_frames, args, fps=25)

if __name__ == '__main__':
    main()
