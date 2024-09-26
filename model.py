import torch 
from PIL import Image
from torchvision import transforms

def load_model():
    global model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

def model_prob(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.5,2.0)),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    
    with torch.no_grad():
        output = model(input_batch)

    return torch.nn.functional.softmax(output[0], dim=0)

def model_pred(probabilities):
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
# Show top categories per image
    top_prob, top_catid = torch.topk(probabilities, 1)
    return categories[top_catid[0]]

def main():
    load_model()
    print("This is our model :", model)
    filename ="dog.jpg"
    probabilities=model_prob(filename=filename)
   # print(probabilities)
    print(model_pred(probabilities))


if __name__=="__main__":
    main()
    