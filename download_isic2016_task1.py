
"""
Download ISIC 2016 Task 1 Dataset
- Training images (900 images)
- Training ground truth segmentation masks (900 masks)
- Test images (379 images)
- Test ground truth segmentation masks (379 masks)
"""

import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Base API URL
API_BASE = "https://api.isic-archive.com/api/v2"

# Collection IDs
TRAINING_COLLECTION_ID = 74  # Challenge 2016: Training (900 images)
TEST_COLLECTION_ID = 61      # Challenge 2016: Test (379 images)

# Output directories
OUTPUT_DIR = Path("D:/project 2/data/ISIC2016_Task1")
TRAIN_IMAGES_DIR = OUTPUT_DIR / "ISBI2016_ISIC_Part1_Training_Data"
TRAIN_GT_DIR = OUTPUT_DIR / "ISBI2016_ISIC_Part1_Training_GroundTruth"
TEST_IMAGES_DIR = OUTPUT_DIR / "ISBI2016_ISIC_Part1_Test_Data"
TEST_GT_DIR = OUTPUT_DIR / "ISBI2016_ISIC_Part1_Test_GroundTruth"


def create_directories():
    """Create output directories."""
    for dir_path in [TRAIN_IMAGES_DIR, TRAIN_GT_DIR, TEST_IMAGES_DIR, TEST_GT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directories in: {OUTPUT_DIR}")


def get_image_ids(collection_id):
    """Get all image IDs from a collection."""
    image_ids = []
    page = 1
    
    while True:
        url = f"{API_BASE}/images/?collections={collection_id}&page={page}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for result in data.get('results', []):
                image_ids.append(result['isic_id'])
            
            # Check if there are more pages
            if not data.get('results') or len(data.get('results', [])) < 50:
                break
            page += 1
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    return image_ids


def download_image(image_id, output_dir, file_suffix=""):
    """Download a single image."""
    image_url = f"https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"
    output_path = output_dir / f"{image_id}{file_suffix}.jpg"
    
    if output_path.exists():
        return f"Skipped {image_id} (already exists)"
    
    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return f"Downloaded {image_id}.jpg"
    except Exception as e:
        return f"Error downloading {image_id}: {e}"


def download_segmentation_mask(image_id, output_dir):
    """Download segmentation mask for an image."""
    # Ground truth masks are stored with _Segmentation suffix
    mask_url = f"https://isic-archive.s3.amazonaws.com/segmentations/{image_id}_segmentation.png"
    output_path = output_dir / f"{image_id}_segmentation.png"
    
    if output_path.exists():
        return f"Skipped {image_id} mask (already exists)"
    
    try:
        response = requests.get(mask_url, timeout=60)
        if response.status_code == 404:
            # Try alternative URL format
            mask_url_alt = f"https://isic-archive.s3.amazonaws.com/masks/{image_id}_segmentation.png"
            response = requests.get(mask_url_alt, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return f"Downloaded {image_id}_segmentation.png"
    except Exception as e:
        return f"Error downloading mask for {image_id}: {e}"


def download_collection_images(collection_id, images_dir, gt_dir, collection_name):
    """Download all images and ground truth for a collection."""
    print(f"\n{'='*60}")
    print(f"Downloading {collection_name}")
    print(f"{'='*60}")
    
    # Get image IDs
    print(f"Fetching image list from collection {collection_id}...")
    image_ids = get_image_ids(collection_id)
    print(f"Found {len(image_ids)} images")
    
    # Download images
    print(f"\nDownloading images to {images_dir}...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_image, img_id, images_dir): img_id for img_id in image_ids}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(image_ids)} images")
    
    print(f"Downloaded images to {images_dir}")
    
    # Download ground truth masks
    print(f"\nDownloading segmentation masks to {gt_dir}...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_segmentation_mask, img_id, gt_dir): img_id for img_id in image_ids}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(image_ids)} masks")
    
    print(f"Downloaded masks to {gt_dir}")


def main():
    print("="*60)
    print("ISIC 2016 Task 1 Dataset Downloader")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Download Training data
    download_collection_images(
        TRAINING_COLLECTION_ID,
        TRAIN_IMAGES_DIR,
        TRAIN_GT_DIR,
        "ISIC 2016 Training Data"
    )
    
    # Download Test data
    download_collection_images(
        TEST_COLLECTION_ID,
        TEST_IMAGES_DIR,
        TEST_GT_DIR,
        "ISIC 2016 Test Data"
    )
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  Training images: {TRAIN_IMAGES_DIR}")
    print(f"  Training masks:  {TRAIN_GT_DIR}")
    print(f"  Test images:     {TEST_IMAGES_DIR}")
    print(f"  Test masks:      {TEST_GT_DIR}")


if __name__ == "__main__":
    main()
