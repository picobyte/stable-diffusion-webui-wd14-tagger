from typing import Dict
from math import ceil
from json import load
import argparse
import re
from collections import defaultdict

# read two json files, compare the weighted frequencies of the tags in the two
# files the first file is json and contains all safetensor files, with major
# sections and weighted tags

# the second file is the result of an interrogations of images, with weighted
# tags. tags may be substrings of the tags in the first file

# next argument is an interrogator id, a string
# optinally there are comma delimited images as arguments

# in the end print the top ten safetensors and major sections that containss
# the tags that are most similar to the tags in the second file

# all weights are between 0 and 1, higher is more important


# example usage:
# first run shell_scripts/create_safetensors_db.sh
# then interrogate an image in a subdirectory test/


# cd stable-diffusion-webui/extensions/stable-diffusion-webui-wd14-tagger/
#
# python shell_scripts/compare_weighted_frequencies.py safetensors_db.json \
#        test/db.json

# # .. lists used interrogation models


# python shell_scripts/compare_weighted_frequencies.py safetensors_db.json \
#        -c 20 test/db.json <model>


desc = 'Compare weighted frequencies of tags in two json file'
parser = argparse.ArgumentParser(description=desc)
hlp = 'number of results to print'
parser.add_argument('-c', '--count', default=10, type=int, help=hlp)
parser.add_argument('file1', help='all safetensors json file')
parser.add_argument('file2', help='image interrogation json file')
parser.add_argument('id', help='interrogator id', nargs='?', default="")
parser.add_argument('images', nargs='*', help='images', default=[])
args = parser.parse_args()


with open(args.file1) as f:
    all_sftns = load(f)

with open(args.file2) as f:
    data = load(f)

query = data["query"]

indices = set()
if args.id == "":
    uniq = set()
    for k in data["query"]:
        if k not in uniq:
            uniq.add(k[64:])
    if len(uniq) != 1:
        print("Missing interrogator id, contained are:")
        for k in uniq:
            print(k)
        exit(1)
    else:
        # use the only one
        args.id = uniq.pop()

for k, t in data["query"].items():
    img_fn, idx = t
    if k[64:] == args.id:
        if len(args.images) > 0:
            for i in args.images:
                if img_fn[-len(i):] == i:
                    break
            else:
                continue
        indices.add(int(idx))

interrogation_result = {}
for t, lst in data["tag"].items():
    wt = 0.0
    for stored in lst:
        i = ceil(stored) - 1
        if i in indices:
            wt += stored - i
    if wt > 0.0:
        interrogation_result[t] = wt / len(indices)

scores: Dict[str, float] = defaultdict(float)

for safetensor in all_sftns:
    for major in all_sftns[safetensor]:
        ct = len(all_sftns[safetensor][major])
        if ct == 0:
            continue

        for tag, wt in interrogation_result.items():
            if tag in all_sftns[safetensor][major]:
                sftns_wt = all_sftns[safetensor][major][tag]
                n = (1.0 - abs(sftns_wt - wt))
                scores[safetensor + "\t" + major] += n / ct
            else:
                rex = re.compile(r'\b{}\b'.format(tag))
                t_len = len(tag)
                # the tag may be a substring of a tag in the safetensor
                # however only entire words are considered and a penalty if the
                # string lenghts are close to each other
                highest = 0.0
                for sftns_tag in all_sftns[safetensor][major]:
                    if rex.search(sftns_tag):
                        sftns_tag_len = len(sftns_tag)
                        sftns_wt = all_sftns[safetensor][major][sftns_tag]
                        n = (sftns_tag_len - t_len) / sftns_tag_len
                        n -= abs(sftns_wt - wt)
                        highest = max(highest, n)
                scores[safetensor + "\t" + major] += highest / ct

# sort the scores
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# print the top ten safetensors and major sections
for i in range(args.count):
    if i >= len(sorted_scores):
        break
    print(sorted_scores[i][0] + "\t" + str(sorted_scores[i][1]))
