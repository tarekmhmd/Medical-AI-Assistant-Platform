#!/bin/bash

echo "í³¦ Creating folders..."
mkdir -p data/skin data/sound data/lab

echo "í·ª Downloading lab dataset (~5MB)..."
kaggle datasets download -d prasad22/healthcare-dataset -p data/lab --unzip

echo "í¾§ Downloading respiratory sounds (~1GB)..."
kaggle datasets download -d vbookshelf/respiratory-sound-database -p data/sound --unzip

echo "í¹º Downloading skin dataset (~3GB)..."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/skin --unzip

echo "âœ… Small datasets download completed!"
du -sh data/*
