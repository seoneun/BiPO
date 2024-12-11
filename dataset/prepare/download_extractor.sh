rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading extractors"
gdown --fuzzy https://drive.google.com/file/d/19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM/view?usp=sharing

unzip humanml3d_evaluator.zip

echo -e "Cleaning\n"
rm humanml3d_evaluator.zip

echo -e "Downloading done!"
