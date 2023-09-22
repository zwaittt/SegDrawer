from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware

from segment_anything_hq import SamAutomaticMaskGenerator as HqSamAutomaticMaskGenerator, sam_model_registry as Hq_sam_model_registry, SamPredictor as HqSamPredictor
import torch
import os
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode

from pydantic import BaseModel, Field

from aiohttp import ClientSession

class SamRequestDto(BaseModel):
    imageUrl: str = Field(..., description="URL of the image to do segmentation")

class SamResponseDto(BaseModel):
    message: str = Field(..., description="Message from the server")
    image_embedding: str = Field(None, description="Base64 of image embeddings")
    interm_embedding: str = Field(None, description="Base64 of image interm embeddings")

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def np_to_base64(arr):
    base64_str = b64encode(arr.tobytes()).decode('utf-8')
    return base64_str

def get_pt_path(model_type: str):
    pth = 'assets/pt/sam_hq_{}.pth'.format(model_type)
    if os.path.isfile(pth):
        return pth
    return 'assets/pt/sam_hq_vit_l.pth'

async def download_img_content(url: str):
    async with ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
    return content

preset_model_type = os.getenv('MODEL_TYPE')
model_type = preset_model_type if preset_model_type else "vit_l"  # model type
sam_checkpoint = get_pt_path(model_type) # pretrained


if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

print("Loading model", model_type)

sam = Hq_sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

predictor = HqSamPredictor(sam)
mask_generator = HqSamAutomaticMaskGenerator(sam)
print("Finishing loading")

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(GZipMiddleware, minimum_size=1000)

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

@app.post("/image-file", response_model=SamResponseDto)
async def process_image_file(
    image: UploadFile = File(...)
):
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data).convert('RGBA'))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    predictor.set_image(GLOBAL_IMAGE)

    image_embedding_tensor = predictor.get_image_embedding()
    image_embedding = image_embedding_tensor.cpu().numpy()

    vit_features = predictor.interm_features[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
    hq_features = predictor.model.mask_decoder.embedding_encoder(image_embedding_tensor) + predictor.model.mask_decoder.compress_vit_feat(vit_features)
    interm_embedding = hq_features.detach().cpu().numpy()

    print("image embedding", image_embedding.shape)
    print("interm_embedding", interm_embedding.shape)
    
    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
            "image_embedding": np_to_base64(image_embedding.reshape(-1)),
            "interm_embedding": np_to_base64(interm_embedding.reshape(-1)),
        },
        status_code=200,
    )

@app.post("/image-url", response_model=SamResponseDto)
async def process_image_url(
    data: SamRequestDto,
):
    if data.imageUrl is None:
        return JSONResponse(
            content={
                "message": "No image url provided",
            },
            status_code=400,
        )
    
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    # Read the image and mask data as bytes
    image_data = await download_img_content(data.imageUrl)

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data).convert('RGBA'))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    predictor.set_image(GLOBAL_IMAGE)

    image_embedding_tensor = predictor.get_image_embedding()
    image_embedding = image_embedding_tensor.cpu().numpy()

    vit_features = predictor.interm_features[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
    hq_features = predictor.model.mask_decoder.embedding_encoder(image_embedding_tensor) + predictor.model.mask_decoder.compress_vit_feat(vit_features)
    interm_embedding = hq_features.detach().cpu().numpy()

    print("image embedding", image_embedding.shape)
    print("interm_embedding", interm_embedding.shape)
    
    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
            "image_embedding": np_to_base64(image_embedding.reshape(-1)),
            "interm_embedding": np_to_base64(interm_embedding.reshape(-1)),
        },
        status_code=200,
    )

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=80)