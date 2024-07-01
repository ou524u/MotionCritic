cd MotionCritic
# prepare smpl files
mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip
cd ..

# prepare rendering for fine-tuning(optional)

# cd MDMCritic
# # prepare smpl files
# mkdir -p body_models
# cd body_models/

# echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
# gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
# rm -rf smpl

# unzip smpl.zip
# echo -e "Cleaning\n"
# rm smpl.zip