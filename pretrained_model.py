import os
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def load_dataset_config():
    if os.path.exists('data.yaml'):
        with open('data.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config
    else:
        config = {
            'path': '.',
            'train': 'data/train',
            'val': 'data/val',
            'test': 'data/test',
            'names': ['smoke', 'fire']
        }

        with open('data.yaml', 'w') as file:
            yaml.dump(config, file)

        return config

def train_model(epochs=10, img_size=640, batch_size=16, pretrained_weights=None):
    if pretrained_weights:
        model = YOLO(pretrained_weights)
    else:
        model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,
        save=True,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    return model

def evaluate_model(model):

    results = model.val()
    
    metrics = results.box
    print(f"mAP50: {metrics.map50:.4f}")
    print(f"mAP50-95: {metrics.map:.4f}")
    print(f"Precision: {metrics.p:.4f}")
    print(f"Recall: {metrics.r:.4f}")
    
    return results

def test_on_images(model, test_dir='data/test/images', num_samples=5):
    images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) 
              if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) > num_samples:
        images = np.random.choice(images, num_samples, replace=False)
    
    fig, axes = plt.subplots(len(images), 2, figsize=(12, 4*len(images)))
    if len(images) == 1:
        axes = [axes]
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        axes[i][0].imshow(np.array(img))
        axes[i][0].set_title('Imagem Original')
        axes[i][0].axis('off')

        results = model.predict(img_path, conf=0.25)
        result_img = results[0].plot()
        axes[i][1].imshow(result_img)
        axes[i][1].set_title('Detecções')
        axes[i][1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()
    
    return results

def export_model(model, format='onnx'):
    model.export(format=format)
    print(f"Modelo exportado no formato {format}")

def main():
    config = load_dataset_config()
    print("Configuração do dataset carregada:")
    print(config)
    
    for split in ['train', 'val', 'test']:
        data_dir = config.get(split, f'data/{split}')
        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"AVISO: Diretório {img_dir} ou {label_dir} não encontrado!")
    
    print("Iniciando treinamento do modelo...")
    model = train_model(epochs=10)
    
    print("Avaliando modelo...")
    evaluate_model(model)
    
    print("Testando modelo em imagens...")
    test_on_images(model)
    
    print("Exportando modelo...")
    export_model(model)
    
    print("Processo completo!")

if __name__ == "__main__":
    main() 