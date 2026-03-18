from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import StreamingResponse
import io
from PIL import Image, ImageDraw
from utils import load_model, predict
from fastapi.middleware.cors import CORSMiddleware
from utils import get_EfficientNet,get_denseNet,get_resNet,cam_Overlay
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "model-1": load_model(get_resNet(),"asd_model.pth"),
    "model-2": load_model(get_denseNet(),"asd_model_denseNet.pth"),
    "model-3": load_model(get_EfficientNet(),"asd_model_efficientNet.pth"),
}

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    print(model_name)
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")

    model = models[model_name]

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)) 
    img.save("hello.png", "PNG")
    output, image = predict(model, image_bytes)
    predicted_label = output.argmax().item()
    result="Autistic" if not predicted_label else "Non_Autistic"
    cam_Overlay(model_name)

    def encode_image(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    image2=encode_image(Image.open("cam_overlay.png"))

    return {"images":image2,"result":result}

