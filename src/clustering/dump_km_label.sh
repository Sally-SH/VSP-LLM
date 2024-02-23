#!/bin/bash


cluster_nums=("20" "50" "100" "200" "500" "1000" "2000")
rank_nums=("0")

for cluster in "${cluster_nums[@]}"
do
    for rank in "${rank_nums[@]}"
    do
        python dump_km_label.py /mnt/ssd/Dataset/lrs3/av_feat/ train /mnt/ssd/Dataset/lrs3/av_feat/new/$cluster.km 4 $rank /mnt/ssd/jh/Exp/acl24/unit_qlora/av_based/433h/$cluster
    done
done

# cluster_nums=("20" "50" "100" "200" "500" "1000" "2000")

# for cluster in "${cluster_nums[@]}"
# do

#     python dump_km_label.py /mnt/ssd/Dataset/lrs3/av_feat/ test /mnt/ssd/Dataset/lrs3/av_feat/new/$cluster.km 1 0 /mnt/ssd/jh/Exp/acl24/unit_qlora/av_based/433h/$cluster

# done