import pathlib
import sys
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import torch
from torchvision import models
from torchvision import transforms


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


def print_details(name, image_path, labels, resnet101):
    label, confidence = predict(image_path, labels, resnet101)
    print("{} is identified as : {}\twith confidence : {}".format(name, label, confidence))
    print()

test_example_dog = pathlib.Path("./../../images/dog.jpg")
test_example_fruit = pathlib.Path("./../../images/fruit.jpg")
test_example_car = pathlib.Path("./../../images/car.jpg")

if __name__ == '__main__':
    CACHE_DIR = pathlib.Path("./../classification_model_pretrained_weights")
    torch.hub.set_dir(CACHE_DIR)

    available_models: list[str] = dir(models)

    resnet101 = None
    for model in available_models:
        if 'resnet101' in model.lower():
            resnet101 = models.resnet101(weights=True)
            break

    if resnet101 is None:
        print("[ERROR]: No model named 'alexnet' present among the available models!")
        sys.exit(1)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    image_true_labels = {
        'Dog': test_example_dog,
        'Strawberry': test_example_fruit,
        'Car': test_example_car
    }
    for t_label in image_true_labels:
        print_details(t_label, image_true_labels[t_label], labels, resnet101)