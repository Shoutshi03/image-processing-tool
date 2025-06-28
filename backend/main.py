from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from skimage import filters, morphology, segmentation, feature
from skimage.filters import threshold_otsu
from skimage.segmentation import slic
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from typing import Optional
import json

app = FastAPI(title="Image Processing Tool API", version="1.0.0")

# Configuration CORS pour permettre les requêtes depuis le frontend Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(image):
    """Convertit une image numpy en base64"""
    if len(image.shape) == 3:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = Image.fromarray(image)
    
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    """Convertit une chaîne base64 en image numpy"""
    img_data = base64.b64decode(base64_str)
    img_pil = Image.open(io.BytesIO(img_data))
    img_array = np.array(img_pil)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

@app.get("/")
async def root():
    return {"message": "Image Processing Tool API"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Télécharge une image et retourne sa représentation base64"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Format d'image non supporté")
        
        # Convertir en base64
        img_base64 = image_to_base64(image)
        
        return {
            "success": True,
            "image": img_base64,
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if len(image.shape) == 3 else 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/convert-grayscale")
async def convert_grayscale(image_data: str = Form(...)):
    """Convertit une image en niveaux de gris"""
    try:
        image = base64_to_image(image_data)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result_base64 = image_to_base64(gray_image)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la conversion: {str(e)}")

@app.post("/histogram")
async def generate_histogram(image_data: str = Form(...)):
    """Génère l'histogramme d'une image"""
    try:
        image = base64_to_image(image_data)
        
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Calculer l'histogramme
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        plt.plot(hist)
        plt.title('Histogramme d\'intensité')
        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')
        plt.grid(True)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buffer.seek(0)
        hist_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "histogram": hist_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'histogramme: {str(e)}")

@app.post("/flip")
async def flip_image(image_data: str = Form(...), direction: str = Form(...)):
    """Effectue un retournement horizontal ou vertical"""
    try:
        image = base64_to_image(image_data)
        
        if direction == "horizontal":
            flipped = cv2.flip(image, 1)
        elif direction == "vertical":
            flipped = cv2.flip(image, 0)
        else:
            raise HTTPException(status_code=400, detail="Direction invalide. Utilisez 'horizontal' ou 'vertical'")
        
        result_base64 = image_to_base64(flipped)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du retournement: {str(e)}")

@app.post("/rotate")
async def rotate_image(image_data: str = Form(...), angle: float = Form(...)):
    """Effectue une rotation de l'image"""
    try:
        image = base64_to_image(image_data)
        
        # Obtenir les dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Appliquer la rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        result_base64 = image_to_base64(rotated)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la rotation: {str(e)}")

@app.post("/resize")
async def resize_image(image_data: str = Form(...), width: int = Form(...), height: int = Form(...)):
    """Redimensionne une image"""
    try:
        image = base64_to_image(image_data)
        
        # Redimensionner
        resized = cv2.resize(image, (width, height))
        
        result_base64 = image_to_base64(resized)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du redimensionnement: {str(e)}")

@app.post("/threshold")
async def threshold_image(image_data: str = Form(...), method: str = Form(...), threshold_value: Optional[int] = Form(None)):
    """Applique un seuillage global ou adaptatif"""
    try:
        image = base64_to_image(image_data)
        
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        if method == "global":
            if threshold_value is None:
                # Seuillage automatique avec Otsu
                threshold_value, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "adaptive":
            thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            raise HTTPException(status_code=400, detail="Méthode invalide. Utilisez 'global' ou 'adaptive'")
        
        result_base64 = image_to_base64(thresholded)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du seuillage: {str(e)}")

@app.post("/denoise")
async def denoise_image(image_data: str = Form(...), method: str = Form(...), kernel_size: Optional[int] = Form(5)):
    """Applique une réduction du bruit"""
    try:
        image = base64_to_image(image_data)
        
        if method == "gaussian":
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "median":
            denoised = cv2.medianBlur(image, kernel_size)
        else:
            raise HTTPException(status_code=400, detail="Méthode invalide. Utilisez 'gaussian' ou 'median'")
        
        result_base64 = image_to_base64(denoised)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la réduction du bruit: {str(e)}")

@app.post("/morphology")
async def morphology_operation(image_data: str = Form(...), operation: str = Form(...), kernel_size: Optional[int] = Form(5)):
    """Applique des opérations morphologiques"""
    try:
        image = base64_to_image(image_data)
        
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Créer le noyau
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == "erosion":
            result = cv2.erode(gray_image, kernel, iterations=1)
        elif operation == "dilation":
            result = cv2.dilate(gray_image, kernel, iterations=1)
        elif operation == "opening":
            result = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        elif operation == "closing":
            result = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        else:
            raise HTTPException(status_code=400, detail="Opération invalide. Utilisez 'erosion', 'dilation', 'opening' ou 'closing'")
        
        result_base64 = image_to_base64(result)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'opération morphologique: {str(e)}")

@app.post("/edge-detection")
async def edge_detection(image_data: str = Form(...), method: str = Form(...)):
    """Détection de contours"""
    try:
        image = base64_to_image(image_data)
        
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        if method == "canny":
            edges = cv2.Canny(gray_image, 50, 150)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        else:
            raise HTTPException(status_code=400, detail="Méthode invalide. Utilisez 'canny' ou 'sobel'")
        
        result_base64 = image_to_base64(edges)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection de contours: {str(e)}")

@app.post("/superpixels")
async def superpixel_segmentation(image_data: str = Form(...), n_segments: Optional[int] = Form(100)):
    """Segmentation par superpixels"""
    try:
        image = base64_to_image(image_data)
        
        # Convertir BGR vers RGB pour scikit-image
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Appliquer SLIC
        segments = slic(rgb_image, n_segments=n_segments, compactness=10, sigma=1)
        
        # Marquer les frontières
        from skimage.segmentation import mark_boundaries
        marked = mark_boundaries(rgb_image, segments)
        
        # Convertir en uint8
        marked_uint8 = img_as_ubyte(marked)
        
        # Reconvertir en BGR pour OpenCV
        if len(marked_uint8.shape) == 3:
            marked_bgr = cv2.cvtColor(marked_uint8, cv2.COLOR_RGB2BGR)
        else:
            marked_bgr = marked_uint8
        
        result_base64 = image_to_base64(marked_bgr)
        
        return {
            "success": True,
            "result": result_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la segmentation: {str(e)}")

@app.post("/face-detection")
async def face_detection(image_data: str = Form(...)):
    """Détection de visages"""
    try:
        image = base64_to_image(image_data)
        
        # Charger le classificateur de visages
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convertir en niveaux de gris pour la détection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Dessiner des rectangles autour des visages
        result_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        result_base64 = image_to_base64(result_image)
        
        return {
            "success": True,
            "result": result_base64,
            "faces_count": len(faces)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection de visages: {str(e)}")

@app.post("/face-anonymization")
async def face_anonymization(image_data: str = Form(...), method: str = Form(...)):
    """Anonymisation des visages"""
    try:
        image = base64_to_image(image_data)
        
        # Charger le classificateur de visages
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convertir en niveaux de gris pour la détection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Appliquer l'anonymisation
        result_image = image.copy()
        for (x, y, w, h) in faces:
            if method == "blur":
                # Appliquer un flou gaussien
                face_region = result_image[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                result_image[y:y+h, x:x+w] = blurred_face
            elif method == "pixelate":
                # Pixeliser le visage
                face_region = result_image[y:y+h, x:x+w]
                # Réduire la taille puis agrandir pour créer l'effet pixelisé
                small = cv2.resize(face_region, (w//10, h//10))
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result_image[y:y+h, x:x+w] = pixelated
            else:
                raise HTTPException(status_code=400, detail="Méthode invalide. Utilisez 'blur' ou 'pixelate'")
        
        result_base64 = image_to_base64(result_image)
        
        return {
            "success": True,
            "result": result_base64,
            "faces_anonymized": len(faces)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'anonymisation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

