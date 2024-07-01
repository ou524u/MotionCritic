cd MotionCritic
mkdir -p datasets
cd datasets
gdown https://drive.google.com/uc?id=1H5MAPBIAygGV5HSa2yIftWDdGq4fPEXB
unzip mlists.zip
rm -f mlists.zip

cd ..
mkdir -p marked
cd marked
# download organized annotations.
gdown https://drive.google.com/uc?id=1Lgg_ccVvAfxvH0UF-w3Z5OI2rvdFy53V
unzip annotations-organized.zip
rm -f annotations-organized.zip

