import os
import shutil
import pandas as pd
from tqdm import tqdm

base_dir = 'data'
output_dir = 'processed_data'

for dataset_type in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, dataset_type, 'fire'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_type, 'smoke'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_type, 'none'), exist_ok=True)

metadata = []

for dataset_type in ['train', 'val', 'test']:
    images_dir = os.path.join(base_dir, dataset_type, 'images')
    labels_dir = os.path.join(base_dir, dataset_type, 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Diretório {dataset_type} não encontrado. Pulando...")
        continue

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc=f"Processando {dataset_type}"):
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
            if not content:
                target_class = 'none'
            else:
                lines = content.split('\n')
                classes = set()

                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            classes.add(int(parts[0]))

                if 0 in classes and 1 in classes:
                    target_class = 'fire'
                elif 1 in classes:
                    target_class = 'fire'
                elif 0 in classes:
                    target_class = 'smoke'
                else:
                    target_class = 'none'
        else:
            target_class = 'none'

        target_path = os.path.join(output_dir, dataset_type, target_class, img_file)
        shutil.copy(img_path, target_path)

        metadata.append({
            'original_path': img_path,
            'new_path': target_path,
            'class': target_class,
            'dataset_type': dataset_type
        })

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

print(f"Processamento concluído. Dados organizados em {output_dir}")

print("\nEstatísticas do dataset:")
for dataset_type in ['train', 'val', 'test']:
    subset = metadata_df[metadata_df['dataset_type'] == dataset_type]
    if not subset.empty:
        print(f"\n{dataset_type.upper()}:")
        class_counts = subset['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} imagens")
