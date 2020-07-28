if [ -d data ]; then
    echo "path ./data is already exists"
else
    mkdir data
    mkdir data/pretrained
    wget http://aoi.naist.jp/MedEXJ2/pretrained/final.model -O data/pretrained/final.model
    wget http://aoi.naist.jp/MedEXJ2/pretrained/labels.txt -O data/pretrained/labels.txt
    wget http://aoi.naist.jp/MedEXJ2/norm_dic.csv -O data/norm_dic.csv
fi
