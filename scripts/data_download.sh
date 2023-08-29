# Please run this shell in NewsRecommendation root_dir
mkdir data && cd data

# Glove
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Demo
# mkdir MINDdemo && cd MINDdemo
# wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip
# unzip MINDdemo_train.zip -d train
# unzip MINDdemo_dev.zip -d val
# cp -r val test
# rm MINDdemo_*.zip
# cd ..


# Small
mkdir MINDsmall && cd MINDsmall
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test
rm MINDsmall_*.zip
cd ..

# Large
mkdir MINDlarge && cd MINDlarge
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_train.zip -d train
unzip MINDlarge_dev.zip -d val
unzip MINDlarge_test.zip -d test
rm MINDlarge_*.zip
cd ..

