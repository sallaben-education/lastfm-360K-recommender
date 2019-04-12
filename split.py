#!/Users/sallaben/anaconda3/bin/python
splitLen = 1000000         # lines per file
outputBase = 'lastfm-dataset-360K/o'  # o.1.tsv, o.2.tsv, etc.

input = open('lastfm-dataset-360K/obsession3.tsv', 'r')

count = 0
at = 0
dest = None
for line in input:
    if count % splitLen == 0:
        if dest: dest.close()
        dest = open(outputBase + str(at) + '.tsv', 'w')
        at += 1
    dest.write(line)
    count += 1
