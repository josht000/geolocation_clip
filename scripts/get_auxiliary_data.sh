# Downloads data needed for data augmentation and training on auxiliary data.
# Ubuntu 22.04 has bunzip2 installed by default, but not unzip.

sudo apt-get install -y unzip

# OSV-Mini-129k
mkdir -p data/osv-mini-129k && cd data/osv-mini-129k
wget https://www.kaggle.com/api/v1/datasets/download/josht000/osv-mini-129k -O osv-mini-129k.zip
unzip osv-mini-129k.zip
mv osv5m/* ./ && rm -r osv5m && rm osv-mini-129k.zip && cd ../..
