import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import torchvision.models as models
from torchvision import models, transforms
from PIL import Image
import torch
from Grad_Cm import denormalize,find_last_conv_layer,IMAGENET_MEAN,IMAGENET_STD,GradCAMPlusPlus,overlay_cam_on_image



def get_resNet():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2) 
    return model

def get_denseNet():
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def get_EfficientNet():
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    return model

def load_model(model,model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mod = {
    "model-1": load_model(get_resNet(),"asd_model.pth"),
    "model-2": load_model(get_denseNet(),"asd_model_denseNet.pth"),
    "model-3": load_model(get_EfficientNet(),"asd_model_efficientNet.pth"),
}

def predict(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    return output, image

def cam_Overlay(model_name):
    model=mod[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()


    target_layer = find_last_conv_layer(model)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_path = "hello.png"
    pil_img = Image.open(img_path).convert("RGB")
    inp = preprocess(pil_img).unsqueeze(0)


    cam_engine =GradCAMPlusPlus(model, target_layer)
    cam_01, pred_idx = cam_engine.generate(inp, class_idx=None)
    print("Predicted class index:", pred_idx)

    img_uint8 = denormalize(inp[0])
    overlay = overlay_cam_on_image(img_uint8, cam_01, alpha=0.40)
    Image.fromarray(overlay).save("cam_overlay.png")
    print("Saved: cam_overlay.jpg")