if [ -d data ]; then
    echo "path ./data is already exists"
else
    mkdir data
    wget http://aoi.naist.jp/MedEXJ2/pretrained -O data/pretrained
    wget http://aoi.naist.jp/MedEXJ2/norm_dic.csv -O data/norm_dic.csv
fi
