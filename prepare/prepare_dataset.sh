cd MotionCritic
mkdir -p datasets
cd datasets
gdown https://drive.google.com/uc?id=1aRR6uTL4UWaLGtd6aOzU7PoHUkbPeIc9
unzip mlists_corrected.zip
rm -f mlists_corrected.zip

cd ..
mkdir -p marked
cd marked
# download organized annotations.
gdown https://drive.google.com/uc?id=1Lgg_ccVvAfxvH0UF-w3Z5OI2rvdFy53V
unzip annotations-organized.zip
rm -f annotations-organized.zip

