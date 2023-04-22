import pathlib
import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

from PIL import Image
import torch
from torchvision import models
from torchvision import transforms

# set ``cache_dir`` for storing pretrained alexnet weights
CACHE_DIR = pathlib.Path("./../classification_model_pretrained_weights")
torch.hub.set_dir(CACHE_DIR)

# some test examples
test_example_dog = pathlib.Path("./../../images/dog.jpg")
test_example_fruit = pathlib.Path("./../../images/fruit.jpg")
test_example_car = pathlib.Path("./../../images/car.jpg")


def cl_parse_arguments():
    """Parse the command line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, default="", help="path to the input image")
    ap.add_argument("-s", "--size", type=str, default="(200, 150)", help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())
    return args

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image, labels, model):
    img = Image.open(image)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Inference using pre-trained ``AlexNet`` model available in ``torchvision.models``
    model.eval()
    out = model(batch_t)
    
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    print(
        [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
    
    return (labels[index[0]], percentage[index[0]].item())


def print_details(name, image_path, labels, alexnet):
    label, confidence = predict(image_path, labels, alexnet)
    print("{} is identified as : {}\twith confidence : {}".format(name, label, confidence))
    print()


if __name__ == '__main__':
    args = cl_parse_arguments()
    
    available_models: list[str] = dir(models)

    alexnet = None
    for model in available_models:
        if 'alexnet' in model.lower():
            alexnet = models.alexnet(pretrained=True)
            break

    if alexnet is None:
        print("[ERROR]: No model named 'alexnet' present among the available models!")
        sys.exit(1)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    if args['image'] == "":
        image_true_labels = {
            'Dog': test_example_dog,
            'Strawberry': test_example_fruit,
            'Car': test_example_car}
        for t_label in image_true_labels:
            print_details(t_label, image_true_labels[t_label], labels, alexnet)
    else:
        file_name = os.path.basename(args['image'])
        t_label = os.path.splitext(file_name)[0]
        print_details(t_label, args['image'], labels, alexnet)