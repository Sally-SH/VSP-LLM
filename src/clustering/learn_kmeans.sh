cluster_nums=("20" "50" "100" "200" "500" "1000" "2000")

for cluster in "${cluster_nums[@]}"
do
    python learn_kmeans.py /mnt/ssd/Dataset/lrs3/av_feat/ train 4 /mnt/ssd/Dataset/lrs3/av_feat/new/$cluster.km $cluster --percent 0.3 
done

