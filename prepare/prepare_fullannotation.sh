# download full annotations, in case you need
cd MotionCritic
cd marked
mkdir -p mdm-full
gdown https://drive.google.com/uc?id=1TpZ0nVvx2c84rYGmHsdLgNbu8gBwLGkA
unzip annotations-mdmfull.zip

rm -f annotations-mdmfull.zip


# flame annotations holds inferior qualities and is not recommended to be used
cd ..
mkdir -p flame-full
gdown https://drive.google.com/uc?id=1mL7aRy2MhrcpAPVNUZ47AVJ-3r8zbN6_
# unzip annotations-flamefull.zip
rm -f annotations-flamefull.zip

