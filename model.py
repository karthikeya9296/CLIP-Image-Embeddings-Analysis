import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set the device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_images(image_dir):
    """Load and preprocess images from a directory."""
    images = []
    for img_name in tqdm(os.listdir(image_dir)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_dir, img_name)
            image = Image.open(path).convert("RGB")
            images.append(processor(images=image, return_tensors="pt").to(device))
    return images

def extract_features(images):
    """Extract image features using CLIP and return as a tensor."""
    with torch.no_grad():
        features = [model.get_image_features(**img) for img in images]
        features = torch.cat(features)
    return features

def perform_tsne(features, n_components=2):
    """Perform t-SNE to reduce dimensionality of features for visualization."""
    tsne = TSNE(n_components=n_components)
    transformed_features = tsne.fit_transform(features.cpu().numpy())
    return transformed_features

def analyze_domain_specific_insights(labeled_data, cluster_labels):
    """Analyze whether images from the same category are grouped together in the clusters."""
    unique_labels = np.unique(labeled_data)
    for label in unique_labels:
        # Extract indices of samples with the current label
        indices = np.where(labeled_data == label)[0]
        # Extract corresponding cluster labels
        cluster_labels_for_label = cluster_labels[indices]
        # Compute the percentage of samples with the same label in each cluster
        cluster_percentage = [np.mean(cluster_labels_for_label == cluster_label) for cluster_label in np.unique(cluster_labels)]
        print(f"Label {label}: {cluster_percentage}")

def introduce_challenging_data_and_observe(tsne_results, cluster_labels):
    """Introduce challenging data (e.g., noisy) and observe its impact on clustering."""
    # For demonstration purposes, let's add some noise to the t-SNE results and observe the impact on clustering
    noisy_tsne_results = tsne_results + np.random.normal(scale=0.1, size=tsne_results.shape)
    plt.figure(figsize=(10, 7))
    plt.scatter(noisy_tsne_results[:, 0], noisy_tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('Noisy t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Cluster Labels')
    plt.show()

# Load images from the specified folder
image_dir = '/Users/karthik/Desktop/GWU/Sem2/Computer Vision/FP/train'  # Update this path to where your images are stored
images = load_images(image_dir)

# Extract features
features = extract_features(images)

# Reduce dimensions and visualize using t-SNE
tsne_results = perform_tsne(features)
plt.figure(figsize=(10, 7))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of Image Embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(tsne_results)
plt.figure(figsize=(10, 7))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.title('Clustered Image Embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# Compute and display the silhouette score for the clustering
silhouette_avg = silhouette_score(tsne_results, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Example usage of domain-specific insights analysis and stress testing
# analyze_domain_specific_insights(labels, kmeans.labels_)
# introduce_challenging_data_and_observe(tsne_results, kmeans.labels_)
