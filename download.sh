DATE=2020-04-24
DATA_DIR=./cord19-"${DATE}"
mkdir "${DATA_DIR}"

wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/comm_use_subset.tar.gz -P "${DATA_DIR}"
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/noncomm_use_subset.tar.gz -P "${DATA_DIR}"
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/custom_license.tar.gz -P "${DATA_DIR}"
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/biorxiv_medrxiv.tar.gz -P "${DATA_DIR}"
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv -P "${DATA_DIR}"
ls "${DATA_DIR}"/*.tar.gz | xargs -I {} tar -zxvf {} -C "${DATA_DIR}"