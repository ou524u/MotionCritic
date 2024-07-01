cd MDMCritic

# prepare a2m dataset
mkdir -p dataset/
cd dataset/

echo "The datasets will be stored in the 'dataset' folder\n"

# HumanAct12 poses
echo "Downloading the HumanAct12 poses dataset"
gdown "https://drive.google.com/uc?id=1130gHSvNyJmii7f6pv5aY5IyQIWc3t7R"
echo "Extracting the HumanAct12 poses dataset"
tar xfzv HumanAct12Poses.tar.gz
echo "Cleaning\n"
rm HumanAct12Poses.tar.gz

# Donwload UESTC poses estimated with VIBE
echo "Downloading the UESTC poses estimated with VIBE"
gdown "https://drive.google.com/uc?id=1LE-EmYNzECU8o7A2DmqDKtqDMucnSJsy"
echo "Extracting the UESTC poses estimated with VIBE"
tar xjvf uestc.tar.bz2
echo "Cleaning\n"
rm uestc.tar.bz2

echo -e "Downloading done!"

# prepare glove for MDM evaluation
cd ..
echo -e "Downloading glove (in use by the evaluators, not by MDM itself)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"

# prepare smpl files
mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"

echo -e "MDM preparations complete !"