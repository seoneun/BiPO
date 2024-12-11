rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading extractors"

gdown --fuzzy https://drive.google.com/file/d/1fiQgD4vQGk6q9ep4CEBWRi_SQAfalwjh/view?usp=sharing

unzip t2m.zip

echo -e "Cleaning\n"
rm t2m.zip

echo -e "Downloading done!"
