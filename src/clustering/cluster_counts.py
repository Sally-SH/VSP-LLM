
unit_pth = ???
units = open(unit_pth).readlines()
count_list = []
for unit_line in units:
    unit_line = unit_line.strip().split(' ')
    int_unit_line = [int(x) for x in unit_line]
    current_count = 1
    counts = []
    for i in range(1, len(int_unit_line)):
        if int_unit_line[i] == int_unit_line[i - 1]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1
    counts.append(current_count)
    str_counts = [str(x) for x in counts]
    count_list.append(' '.join(str_counts) + '\n')
cluster_counts_pth = unit_pth.replace('.km','.cluster_counts')
with open(cluster_counts_pth, 'w') as f:
    f.write(''.join(count_list))