#!/bin/bash
#
# Create a database for safetensors wherein for the
# models each major tag, the occurrence frequency of
# each associated subtag is listed.
#
# requires https://github.com/by321/safetensors_util.git
# gnu parallel, jq, sed, awk
#

# To build the safetensors_db.json database with
# "file.safetensors" { "major tag": { "tag1": <fraction of images>, "tag2": .. } }:


# cd stable-diffusion-webui/extensions/stable-diffusion-webui-wd14-tagger/
# git clone https://github.com/by321/safetensors_util.git
#
# bash shell_scripts/create_safetensors_db.sh -f -p ../../models/Lora -u safetensors_util/ -o safetensors_db.json
#
## now you can compare interrogation weights with the safetensors_db.json using
## shell_scripts/compare_weighted_frequencies.py, see there for usage.


# number of cpus to use by default or use -j to specify
ncpu=$(nproc --all)
[ $ncpu -gt 8 ] && ncpu=8

path=.
utilpath=.
force=0
out=safetensors_db.json

while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        echo "Usage: $0 [-j ncpu] [-p path] [-u utilpath] [-f] [-o out]"
        echo "  -j ncpu      number of cpus to use (default: $ncpu)"
        echo "  -p path      path to directory containing safetensor models (default: ./)"
        echo "  -u utilpath  path to safetensors_util.py"
        echo "  -f           force overwrite of output file"
        echo "  -o out       output file (default: safetensors_db.json)"
        exit 0
        ;;
      -j) ncpu="$2"; shift 2;;
      -p) path="$2/"; shift 2;;
      -u) utilpath="$2/"; shift 2;;
      -f) force=1; shift 1;;
      -o) out="$2"; shift 2;;
    esac
done

if [ ! -d "${path}" ]; then
  echo "Error: '${path}' does not exist (use -p to specify path)"
  exit 1
fi

if [ ! -e "${utilpath}/safetensors_util.py" ]; then
  echo "Error: ${utilpath}/safetensors_util.py does not exist (use -u to specify path)"
  exit 1
fi

if [ -e "${out}" -a $force -eq 0 ]; then
  echo "Error: ${out} already exists (use -f to overwrite)"
  exit 1
fi

ls -1 ${path}/*.safetensors | parallel -n 1 -j $ncpu "python ${utilpath}/safetensors_util.py metadata {} -pm 2>/dev/null |
sed -n '1b;p' | jq -r 'select(.__metadata__ != null) | .__metadata__ | .ss_tag_frequency | select( . != null )' 2>/dev/null | sed 's/\" /\"/' |
awk -v FS=': ' '{
  if (index(\$2, \"null\") > 0) next
  o = index(\$0, \"{\")
  if (o == 1) printf \"\\\"'{}'\\\": \"
  if (o > 0) {
    print \$0
    m = 0
  } else {
    c = index(\$0, \"}\")
    if (c > 0) {
      L=\"\"
      for (i in a) {
        if (L != \"\") print \",\"
        printf \"%s: %.6f\", i, a[i] / m
        L = "x"
      }
      delete a
      if (c == 1) print \$0\",\"
      else print \"\n\"\$0
    } else {
      x = index(\$2, \",\")
      v = int(x != 0 ? substr(\$2, 1, x - 1) : \$2)
      if (v > m) m = v
      a[\$1] = v
    }
  }
}'" | sed -r '
s/^/  /;
1s/^/{\n/;
s/\\"//g
s/^([ \t]+"[^"]+):*(: [01]+(\.[0-9]+)?,?)$/\1"\2/
$s/,?$/\n}/
' > "${out}"


