DATASET=lvis

cd $HOME

mkdir -p  $HOME/$DATASET/train/images
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip -d $HOME/$DATASET/train/images
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip && unzip lvis_v1_train.json.zip
mkdir -p $HOME/$DATASET/val/images
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip -d $HOME/$DATASET/val/images
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip && unzip lvis_v1_val.json.zip

rm train2017.zip
rm val2017.zip
rm lvis_v1_train.json.zip
rm lvis_v1_val.json.zip