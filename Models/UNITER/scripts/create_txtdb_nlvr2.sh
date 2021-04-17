# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


for SPLIT in 'train' 'valid' 'test'; do
    echo "preprocessing ${SPLIT} annotations..."
    python Preprocessing/prepro_nlvr2.py --annotation /data_1/data/uniter/ann/${SPLIT}_si_sp.json \
                         --output /data_1/data/uniter/txt_db/nlvr2_${SPLIT}_si_sp.db
done

echo "done"
