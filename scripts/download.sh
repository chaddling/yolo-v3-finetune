DATASET=lvis

cd $HOME

wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip -d $HOME/$DATASET/train && mv $HOME/$DATASET/train/train2017 $HOME/$DATASET/train/images
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip && unzip lvis_v1_train.json.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip -d $HOME/$DATASET/val && mv $HOME/$DATASET/val/val2017 $HOME/$DATASET/val/images
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip && unzip lvis_v1_val.json.zip

rm train2017.zip
rm val2017.zip
rm lvis_v1_train.json.zip
rm lvis_v1_val.json.zip