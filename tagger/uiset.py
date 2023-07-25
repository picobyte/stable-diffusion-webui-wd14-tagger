""" for handling ui settings """

from typing import List, Dict, Tuple, Callable, Set, Optional
import os
from pathlib import Path
from glob import glob
from math import ceil
from hashlib import sha256
from re import compile as re_comp, sub as re_sub, match as re_match, IGNORECASE
from json import dumps, loads, JSONDecodeError
from jsonschema import validate, ValidationError
from functools import partial
from collections import defaultdict
from PIL import Image

from modules import shared  # pylint: disable=import-error
from modules.deepbooru import re_special  # pylint: disable=import-error
from tagger import format as tags_format  # pylint: disable=import-error
from tagger import settings  # pylint: disable=import-error

Its = settings.InterrogatorSettings

# PIL.Image.registered_extensions() returns only PNG if you call early
supported_extensions = {
    e
    for e, f in Image.registered_extensions().items()
    if f in Image.OPEN
}

# interrogator return type
ItRetTP = Tuple[
    Dict[str, float],  # rating confidences
    Dict[str, float],  # tag confidences
    Dict[str, float],  # excluded tag confidences
    str,               # error message
]


# If the interrogation weights truely have the precision, increase this
# but it should probably stay below 16, or 32 if certified u64, somehow.
# Note that you will have to re-interrogate all images if you change this.
INDEX_SHIFT = 11


class IOData:
    """ data class for input and output paths """
    last_path_mtimes = None
    base_dir = None
    output_root = None
    paths = []
    save_tags = True
    err = set()

    @classmethod
    def error_msg(cls) -> str:
        return "Errors:<ul>" + ''.join(f'<li>{x}</li>' for x in cls.err) + \
               "</ul>"

    @classmethod
    def flip_save_tags(cls) -> callable:
        def toggle():
            cls.save_tags = not cls.save_tags
        return toggle

    @classmethod
    def toggle_save_tags(cls) -> None:
        cls.save_tags = not cls.save_tags

    @classmethod
    def update_output_dir(cls, output_dir: str) -> None:
        """ update output directory, and set input and output paths """
        pout = Path(output_dir)
        if pout != cls.output_root:
            paths = [x[0] for x in cls.paths]
            cls.paths = []
            cls.output_root = pout
            cls.set_batch_io(paths)

    @staticmethod
    def get_bytes_hash(data) -> str:
        """ get sha256 checksum of file """
        # Note: the checksum from an image is not the same as from file
        return sha256(data).hexdigest()

    @classmethod
    def get_hashes(cls) -> Set[str]:
        """ get hashes of all files """
        ret = set()
        for entries in cls.paths:
            if len(entries) == 4:
                ret.add(entries[3])
            else:
                # if there is no checksum, calculate it
                image = Image.open(entries[0])
                checksum = cls.get_bytes_hash(image.tobytes())
                entries.append(checksum)
                ret.add(checksum)
        return ret

    @classmethod
    def update_input_glob(cls, input_glob: str) -> None:
        """ update input glob pattern, and set input and output paths """
        input_glob = input_glob.strip()

        paths = []

        # if there is no glob pattern, insert it automatically
        if not input_glob.endswith('*'):
            if not input_glob.endswith(os.sep):
                input_glob += os.sep
            input_glob += '*'

        # get root directory of input glob pattern
        base_dir = input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)
        msg = 'Invalid input directory'
        if not os.path.isdir(base_dir):
            cls.err.add(msg)
            return
        cls.err.discard(msg)

        recursive = getattr(shared.opts, 'tagger_batch_recursive', '')
        path_mtimes = []
        for filename in glob(input_glob, recursive=recursive):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                path_mtimes.append(os.path.getmtime(filename))
                paths.append(filename)
            elif ext != '.txt' and 'db.json' not in filename:
                print(f'{filename}: not an image extension: "{ext}"')

        # interrogating in a directory with no pics, still flush the cache
        if len(path_mtimes) > 0 and cls.last_path_mtimes == path_mtimes:
            print('No changed images')
            return

        QData.clear(2)
        cls.last_path_mtimes = path_mtimes

        if not cls.output_root:
            cls.output_root = Path(base_dir)
        elif cls.base_dir and cls.output_root == Path(cls.base_dir):
            cls.output_root = Path(base_dir)

        # XXX what is this basedir magic trying to achieve?
        cls.base_dir_last = Path(base_dir).parts[-1]
        cls.base_dir = base_dir

        QData.read_json(cls.output_root)

        print(f'found {len(paths)} image(s)')
        cls.set_batch_io(paths)

    @classmethod
    def set_batch_io(cls, paths: List[str]) -> None:
        """ set input and output paths for batch mode """
        checked_dirs = set()
        cls.paths = []
        for path in paths:
            path = Path(path)
            if not cls.save_tags:
                cls.paths.append([path, '', ''])
                continue

            # guess the output path
            base_dir_last_idx = path.parts.index(cls.base_dir_last)
            # format output filename

            info = tags_format.Info(path, 'txt')
            fmt = partial(lambda info, m: tags_format.parse(m, info), info)

            msg = 'Invalid output format'
            cls.err.discard(msg)
            try:
                formatted_output_filename = tags_format.pattern.sub(
                    fmt,
                    Its.output_filename_format
                )
            except (TypeError, ValueError):
                cls.err.add(msg)

            output_dir = cls.output_root.joinpath(
                *path.parts[base_dir_last_idx + 1:]).parent

            tags_out = output_dir.joinpath(formatted_output_filename)

            if output_dir in checked_dirs:
                cls.paths.append([path, tags_out, ''])
            else:
                checked_dirs.add(output_dir)
                if os.path.exists(output_dir):
                    msg = 'output_dir: not a directory.'
                    if os.path.isdir(output_dir):
                        cls.paths.append([path, tags_out, ''])
                        cls.err.discard(msg)
                    else:
                        cls.err.add(msg)
                else:
                    cls.paths.append([path, tags_out, output_dir])


class QData:
    """ Query data: contains parameters for the query """
    add_tags = []
    keep_tags = set()
    exclude_tags = []
    search_tags = {}
    replace_tags = []
    threshold = 0.35
    tag_frac_threshold = 0.05

    # read from db.json, update with what should be written to db.json:
    json_db = None
    weighed = (defaultdict(list), defaultdict(list))
    query = {}
    index_shift = INDEX_SHIFT

    # representing the (cumulative) current interrogations
    ratings = defaultdict(float)
    tags = defaultdict(list)
    discarded_tags = defaultdict(list)
    in_db = {}
    for_tags_file = defaultdict(lambda: defaultdict(float))

    had_new = False
    err = set()
    image_dups = defaultdict(set)

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str]]:
        def setter(val) -> Tuple[str]:
            setattr(cls, key, val)
        return setter

    @classmethod
    def set_attr(cls, current: str, tag: str) -> None:
        """ set attribute for current tag """
        attr = getattr(cls, current + '_tags')
        if current in ['add', 'replace']:
            attr.append(tag)
        elif current == 'keep':
            attr.add(tag)
        else:
            rex = cls.compile_rex(tag)
            if rex:
                if current == 'exclude':
                    attr.append(rex)
                elif current == 'search':
                    attr[len(attr)] = rex
            else:
                cls.err.add(f'empty regex in {current} tags')

    @classmethod
    def clear(cls, mode: int) -> None:
        """ clear tags and ratings """
        cls.tags.clear()
        cls.discarded_tags.clear()
        cls.ratings.clear()
        cls.for_tags_file.clear()
        if mode > 0:
            cls.in_db.clear()
            cls.image_dups.clear()
        if mode > 1:
            cls.json_db = None
            cls.weighed = (defaultdict(list), defaultdict(list))
            cls.query = {}
        if mode > 2:
            cls.add_tags = []
            cls.keep_tags = set()
            cls.exclude_tags = []
            cls.search_tags = {}
            cls.replace_tags = []

    @classmethod
    def test_add(cls, tag: str, current: str, incompatible: list) -> None:
        """ check if there are incompatible collections """
        msg = f'Empty tag in {current} tags'
        if tag == '':
            cls.err.add(msg)
            return
        cls.err.discard(msg)
        for bad in incompatible:
            if current < bad:
                msg = f'"{tag}" is both in {bad} and {current} tags'
            else:
                msg = f'"{tag}" is both in {current} and {bad} tags'
            attr = getattr(cls, bad + '_tags')
            if bad == 'search':
                for rex in attr.values():
                    if rex.match(tag):
                        cls.err.add(msg)
                        return
            elif bad in 'exclude':
                if any(rex.match(tag) for rex in attr):
                    cls.err.add(msg)
                    return
            else:
                if tag in attr:
                    cls.err.add(msg)
                    return
        cls.set_attr(current, tag)

    @classmethod
    def update_keep(cls, keep: str) -> None:
        cls.keep_tags = set()
        msg = 'Empty tag in keep tags'
        if keep == '':
            cls.err.add(msg)
            return
        cls.err.discard(msg)
        un_re = re_comp(r' keep(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for tag in map(str.strip, keep.split(',')):
            cls.test_add(tag, 'keep', ['exclude', 'search'])

    @classmethod
    def update_add(cls, add: str) -> None:
        cls.add_tags = []
        if add == '':
            return
        un_re = re_comp(r' add(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for tag in map(str.strip, add.split(',')):
            cls.test_add(tag, 'add', ['exclude', 'search'])

        # silently raise count threshold to avoid issue in apply_filters
        count_threshold = getattr(shared.opts, 'tagger_count_threshold', 100)
        if len(cls.add_tags) > count_threshold:
            shared.opts.tagger_count_threshold = len(cls.add_tags)

    @staticmethod
    def compile_rex(rex: str) -> Optional:
        if rex in {'', '^', '$', '^$'}:
            return None
        if rex[0] == '^':
            rex = rex[1:]
        if rex[-1] == '$':
            rex = rex[:-1]
        return re_comp('^'+rex+'$', flags=IGNORECASE)

    @classmethod
    def update_exclude(cls, exclude: str) -> None:
        cls.exclude_tags = []
        if exclude == '':
            return
        un_re = re_comp(r' exclude(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for excl in map(str.strip, exclude.split(',')):
            incompatible = ['add', 'keep', 'search', 'replace']
            cls.test_add(excl, 'exclude', incompatible)

    @classmethod
    def update_search(cls, search_str: str) -> None:
        cls.search_tags = {}
        if search_str == '':
            return
        un_re = re_comp(r' search(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for rex in map(str.strip, search_str.split(',')):
            incompatible = ['add', 'keep', 'exclude', 'replace']
            cls.test_add(rex, 'search', incompatible)

        msg = 'Unequal number of search and replace tags'
        if len(cls.search_tags) != len(cls.replace_tags):
            cls.err.add(msg)
        else:
            cls.err.discard(msg)

    @classmethod
    def update_replace(cls, replace: str) -> None:
        cls.replace_tags = []
        if replace == '':
            return
        un_re = re_comp(r' replace(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for repl in map(str.strip, replace.split(',')):
            cls.test_add(repl, 'replace', ['exclude', 'search'])
        msg = 'Unequal number of search and replace tags'
        if len(cls.search_tags) != len(cls.replace_tags):
            cls.err.add(msg)
        else:
            cls.err.discard(msg)

    @classmethod
    def get_i_wt(cls, val: int) -> Tuple[int, float]:
        """
        in db.json or QData.weighed, the weights & increment in the list are
        encoded. Each filestamp-interrogation corresponds to an incrementing
        index. This index is stored in the upper bits while the weight is
        inverted and stored in the lower bits; masked, so with fixed precision.
        """
        bit_mask = ((1 << cls.index_shift) - 1)
        i = val >> cls.index_shift
        qlen = len(cls.query)
        if i >= qlen:
            raise IndexError(f'Index {i}: out of bounds for query[{qlen}]')
        return i, 1.0 / float(val & bit_mask)

    @classmethod
    def mux_i_wt(cls, i: int, weight: float) -> int:
        """ reverse of get_i_wt """
        bit_mask = ((1 << cls.index_shift) - 1)
        return (i << cls.index_shift) | (int(1.0 / weight) & bit_mask)

    @classmethod
    def read_json(cls, outdir) -> None:
        """ read db.json if it exists, validate it, and update cls """
        cls.json_db = None
        if getattr(shared.opts, 'tagger_auto_serde_json', True):
            cls.json_db = outdir.joinpath('db.json')
            if cls.json_db.is_file():
                print(f'Reading {cls.json_db}')
                cls.had_new = False
                msg = f'Error reading {cls.json_db}'
                cls.err.discard(msg)
                # validate json using either json_schema/db_jon_v1_schema.json
                # or json_schema/db_jon_v2_schema.json

                schema = Path(__file__).parent.parent.joinpath(
                    'json_schema', 'db_json_v1_schema.json'
                )
                try:
                    data = loads(cls.json_db.read_text())
                    validate(data, loads(schema.read_text()))
                    ql = len(data["query"])
                    # convert v1 to v2, after TODO: keep only else branch
                    if "meta" not in data:
                        cls.had_new = True  # <- force write for v1 -> v2
                        for key in ["tag", "rating"]:
                            for tag, lst in data[key].items():
                                new_lst = []
                                for val in lst:
                                    i = ceil(val) - 1
                                    if i >= ql:
                                        raise IndexError(f'{tag}:{i} > {ql}')
                                    new_lst.append(cls.mux_i_wt(i, val - i))
                                data[key][tag] = new_lst
                    else:
                        for key in ["tag", "rating"]:
                            for tag, lst in data[key].items():
                                for val in lst:
                                    val >>= cls.index_shift
                                    if val >= ql:
                                        raise IndexError(f'{tag}:{val} > {ql}')
                except (ValidationError, IndexError) as err:
                    print(f'{msg}: {repr(err)}')
                    cls.err.add(msg)
                    data = {"query": {}, "tag": [], "rating": []}

                cls.index_shift = INDEX_SHIFT
                cls.query = data["query"]
                cls.weighed = (
                    defaultdict(list, data["rating"]),
                    defaultdict(list, data["tag"])
                )
                print(f'Read {cls.json_db}: {len(cls.query)} interrogations, '
                      f'{len(cls.tags)} tags.')

    @classmethod
    def write_json(cls) -> None:
        """ write db.json """
        if cls.json_db is not None:
            data = {
                "rating": cls.weighed[0],
                "tag": cls.weighed[1],
                "query": cls.query,
                "meta": {"index_shift": cls.index_shift}
            }
            cls.json_db.write_text(dumps(data, indent=2))
            print(f'Wrote {cls.json_db}: {len(cls.query)} interrogations, '
                  f'{len(cls.tags)} tags.')

    @classmethod
    def get_index(cls, fi_key: str, path='') -> int:
        """ get index for filestamp-interrogator """
        if path and path != cls.query[fi_key][0]:
            if cls.query[fi_key][0] != '':
                print(f'Dup or rename: Identical checksums for {path}\n'
                      f'and: {cls.query[fi_key][0]} (path updated)')
                cls.had_new = True
            cls.query[fi_key] = (path, cls.query[fi_key][1])

        return cls.query[fi_key][1]

    @classmethod
    def single_data(cls, fi_key: str) -> None:
        """ get tags and ratings for filestamp-interrogator """
        index = cls.query.get(fi_key)[1]
        data = ({}, {})
        for j in range(2):
            for ent, lst in cls.weighed[j].items():
                for i, val in map(cls.get_i_wt, lst):
                    if i == index:
                        data[j][ent] = val
        QData.in_db[index] = ('', '', '') + data

    @classmethod
    def is_excluded(cls, ent: str) -> bool:
        """ check if tag is excluded """
        return any(re_match(x, ent) for x in cls.exclude_tags)

    @classmethod
    def correct_tag(cls, tag: str) -> str:
        """ correct tag for display """
        replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)
        if replace_underscore and tag not in Its.kamojis:
            tag = tag.replace('_', ' ')

        if getattr(shared.opts, 'tagger_escape', False):
            tag = re_special.sub(r'\\\1', tag)  # tag_escape_pattern

        for i, regex in cls.search_tags.items():
            if re_match(regex, tag):
                tag = re_sub(regex, cls.replace_tags[i], tag)
                break

        return tag

    @classmethod
    def apply_filters(cls, data) -> None:
        """ apply filters to query data, store in db.json if required """
        # data = (path, fi_key, tags, ratings, new)
        # fi_key == '' means this is a new file or interrogation for that file

        tags = sorted(data[4].items(), key=lambda x: x[1], reverse=True)

        fi_key = data[2]
        index = len(cls.query)

        ratings = sorted(data[3].items(), key=lambda x: x[1], reverse=True)
        # loop over ratings
        for rating, val in ratings:
            if fi_key != '':
                cls.weighed[0][rating].append(cls.mux_i_wt(index, val))
            cls.ratings[rating] += val

        count_threshold = getattr(shared.opts, 'tagger_count_threshold', 100)
        max_ct = count_threshold - len(cls.add_tags)
        count = 0
        # loop over tags with db update
        for tag, val in tags:
            if isinstance(tag, float):
                print(f'bad return from interrogator, float: {tag} {val}')
                # FIXME: why does this happen? what does it mean?
                continue

            if fi_key != '' and val >= 0.005:
                cls.weighed[1][tag].append(cls.mux_i_wt(index, val))

            if count < max_ct:
                tag = cls.correct_tag(tag)
                if tag not in cls.keep_tags:
                    if cls.is_excluded(tag) or val < cls.threshold:
                        if tag not in cls.add_tags and \
                           len(cls.discarded_tags) < max_ct:
                            cls.discarded_tags[tag].append(val)
                        continue
                if data[1] != '':
                    current = cls.for_tags_file[data[1]].get(tag, 0.0)
                    cls.for_tags_file[data[1]][tag] = min(val + current, 1.0)
                count += 1
            elif fi_key == '':
                break
            if tag not in cls.add_tags:
                # those are already added
                cls.tags[tag].append(val)

        if getattr(shared.opts, 'tagger_verbose', True):
            print(f'{data[0]}: {count}/{len(tags)} tags kept')

        if fi_key != '':
            cls.query[fi_key] = (data[0], index)

    @classmethod
    def finalize_batch(cls, count: int) -> ItRetTP:
        """ finalize the batch query """
        if cls.json_db and cls.had_new:
            cls.write_json()
            cls.had_new = False

        # collect the weights per file/interrogation of the prior in db stored.
        for index in range(2):
            for ent, lst in cls.weighed[index].items():
                for i, val in map(cls.get_i_wt, lst):
                    if i not in cls.in_db:
                        continue
                    cls.in_db[i][3+index][ent] = val

        # process the retrieved from db and add them to the stats
        for got in cls.in_db.values():
            no_floats = sorted(filter(lambda x: not isinstance(x[0], float),
                               got[3].items()), key=lambda x: x[0])
            sorted_tags = ','.join(f'({k},{v:.1f})' for (k, v) in no_floats)
            QData.image_dups[sorted_tags].add(got[0])
            cls.apply_filters(got)

        # average
        return cls.finalize(count)

    @staticmethod
    def sort_tags(tags: Dict[str, float]) -> List[Tuple[str, float]]:
        """ sort tags by value, return list of tuples """
        return sorted(tags.items(), key=lambda x: x[1], reverse=True)

    @classmethod
    def get_image_dups(cls) -> List[str]:
        # first sort values so that those without a comma come first
        ordered = sorted(cls.image_dups.items(), key=lambda x: ',' in x[0])
        return [str(x) for s in ordered if len(s[1]) > 1 for x in s[1]]

    @classmethod
    def finalize(cls, count: int) -> ItRetTP:
        """ finalize the query, return the results """

        count += len(cls.in_db)
        if count == 0:
            return None, None, None, 'no results for query'

        ratings, tags, discarded_tags = {}, {}, {}

        for val in cls.for_tags_file.values():
            for k in cls.add_tags:
                val[k] = 1.0 * count

        for k in cls.add_tags:
            tags[k] = 1.0

        for k, lst in cls.tags.items():
            # len(!) fraction of the all interrogations was above the threshold
            fraction_of_queries = len(lst) / count

            if fraction_of_queries >= cls.tag_frac_threshold:
                # store the average of those interrogations sum(!) / count
                tags[k] = sum(lst) / count
                # trigger an event to place the tag in the active tags list
                # replace if k interferes with html code
            else:
                discarded_tags[k] = sum(lst) / count
                for remaining_tags in cls.for_tags_file.values():
                    if k in remaining_tags:
                        if k not in cls.add_tags and k not in cls.keep_tags:
                            del remaining_tags[k]

        for k, lst in cls.discarded_tags.items():
            fraction_of_queries = len(lst) / count
            discarded_tags[k] = sum(lst) / count

        for ent, val in cls.ratings.items():
            ratings[ent] = val / count

        weighted_tags_files = getattr(shared.opts,
                                      'tagger_weighted_tags_files', False)
        for file, remaining_tags in cls.for_tags_file.items():
            sorted_tags = cls.sort_tags(remaining_tags)
            if weighted_tags_files:
                sorted_tags = [f'({k}:{v})' for k, v in sorted_tags]
            else:
                sorted_tags = [k for k, v in sorted_tags]
            file.write_text(', '.join(sorted_tags), encoding='utf-8')

        warn = ""
        if len(QData.err) > 0:
            warn = "Warnings (fix and try again - it should be cheap):<ul>" + \
                   ''.join([f'<li>{x}</li>' for x in QData.err]) + "</ul>"

        if count > 1 and len(cls.get_image_dups()) > 0:
            warn += "There were duplicates, see gallery tab"
        return ratings, tags, discarded_tags, warn
