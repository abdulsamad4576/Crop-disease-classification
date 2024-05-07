import streamlit as st
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import gdown

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

file_id = '1c1lvtPIKuKzCiLVdVc1zYE1tHN_fbv2c'  
url = f'https://drive.google.com/uc?id={file_id}'
output = 'final.pth'
gdown.download(url, output, quiet=False)

from torchvision import models

model = models.resnext50_32x4d()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5) 
model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))

model.eval()

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)
        if predicted_class==0:
            return 'Cassava Bacterial Blight (CBB)'
        elif predicted_class==1:
            return 'Cassava Brown Streak Disease (CBSD)'
        elif predicted_class==2:
            return 'Cassava Green Mottle (CGM)'
        elif predicted_class==3:
            return 'Cassava Mosaic Disease (CMD)'
        else:
            return 'Healthy'

st.title('Crop Disease Classifier')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    if st.button('Predict'):
        result = predict(image)
        st.write(f"The image is classified as: **{result}**")