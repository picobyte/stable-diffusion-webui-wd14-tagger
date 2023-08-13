""" Interrogator class and subclasses for tagger """
import os
from pathlib import Path
import io
import json
from re import match as re_match
from jsonschema import validate
import inspect
from platform import uname
from typing import Tuple, List, Dict, Callable
from pandas import read_csv
from PIL import Image, UnidentifiedImageError
from numpy import asarray, float32, expand_dims, exp
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from prload import root_dir
from modules import shared
from tagger import settings  # pylint: disable=import-error
from tagger.uiset import QData, IOData  # pylint: disable=import-error
from . import dbimutils  # pylint: disable=import-error # noqa

Its = settings.InterrogatorSettings

# select a device to process
use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

# https://onnxruntime.ai/docs/execution-providers/
# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
onnxrt_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

if shared.cmd_opts.additional_device_ids is not None:
    m = re_match(r'([cg])pu:\d+$', shared.cmd_opts.additional_device_ids)
    if m is None:
        raise ValueError('--device-id is not cpu:<nr> or gpu:<nr>')
    if m.group(1) == 'c':
        onnxrt_providers.pop(0)
    TF_DEVICE_NAME = f'/{shared.cmd_opts.additional_device_ids}'
elif use_cpu:
    TF_DEVICE_NAME = '/cpu:0'
    onnxrt_providers.pop(0)
else:
    TF_DEVICE_NAME = '/gpu:0'

print(f'== WD14 tagger {TF_DEVICE_NAME}, {uname()} ==')


class Interrogator:
    """ Interrogator class for tagger """
    # the raw input and output.
    input = {
        "cumulative": False,
        "large_query": False,
        "unload_after": False,
        "add": '',
        "keep": '',
        "exclude": '',
        "search": '',
        "replace": '',
        "output_dir": '',
    }
    output = None
    odd_increment = 0
    entries = {}

    @classmethod
    def flip(cls, key):
        def toggle():
            cls.input[key] = not cls.input[key]
        return toggle

    @staticmethod
    def get_errors() -> str:
        errors = ''
        if len(IOData.err) > 0:
            # write errors in html pointer list, every error in a <li> tag
            errors = IOData.error_msg()
        if len(QData.err) > 0:
            errors += 'Possible issues:<br><ul><li>' + \
                      '</li><li>'.join(QData.err) + '</li></ul>'
        return errors

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str, str]]:
        def setter(val) -> Tuple[str, str]:
            if key == 'input_glob':
                IOData.update_input_glob(val)
                return (val, cls.get_errors())
            if val != cls.input[key]:
                tgt_cls = IOData if key == 'output_dir' else QData
                getattr(tgt_cls, "update_" + key)(val)
                cls.input[key] = val
            return (cls.input[key], cls.get_errors())

        return setter

    @classmethod
    def refresh(cls) -> List[str]:
        """Refreshes the interrogator entries"""
        if len(cls.entries) == 0:
            it_path = root_dir.joinpath("interrogators.json")
            if not it_path.exists():
                it_path = root_dir.joinpath("default/interrogators.json")
                if not it_path.exists():
                    raise FileNotFoundError(f'{it_path} not found.')

                raw = json.loads(it_path)
                schema = root_dir.joinpath('json_schema',
                                           'interrogators_v1_schema.json')
                validate(raw, json.loads(schema.read_text()))

                for class_name, it in raw.items():
                    if class_name == "DeepDanbooruInterrogator":
                        It_type = DeepDanbooruInterrogator
                    elif class_name == "WaifuDiffusionInterrogator":
                        It_type = WaifuDiffusionInterrogator
                    elif class_name == "MLDanbooruInterrogator":
                        It_type = MLDanbooruInterrogator
                    else:
                        raise ValueError(f'Unimplemented: {it["class"]}')
                    for name, obj in it.items():
                        if name not in obj:
                            obj[name] = name
                        cls.entries[name] = It_type(**obj)

                    cls.entries[name] = It_type(**it["repo_specs"])

    # load deepdanbooru project
        ddp_path = shared.cmd_opts.deepdanbooru_projects_path
        if ddp_path is None:
            ddp_path = Path(shared.models_path, 'deepdanbooru')
        onnx_path = shared.cmd_opts.onnx_path
        if onnx_path is None:
            onnx_path = Path(shared.models_path, 'TaggerOnnx')
        os.makedirs(ddp_path, exist_ok=True)
        os.makedirs(onnx_path, exist_ok=True)

        for path in os.scandir(ddp_path):
            print(f"Scanning {path} as deepdanbooru project")
            if not path.is_dir():
                print(f"Warning: {path} is not a directory, skipped")
                continue

            if not Path(path, 'project.json').is_file():
                print(f"Warning: {path} has no project.json, skipped")
                continue

            cls.entries[path.name] = DeepDanbooruInterrogator(path.name, path)
        # scan for onnx models as well
        for path in os.scandir(onnx_path):
            print(f"Scanning {path} as onnx model")
            if not path.is_dir():
                print(f"Warning: {path} is not a directory, skipped")
                continue

            onnx_files = []
            for file_name in os.scandir(path):
                if file_name.name.endswith('.onnx'):
                    onnx_files.append(file_name)

            if len(onnx_files) != 1:
                print(f"Warning: {path}: multiple .onnx models => skipped")
                continue
            local_path = Path(path, onnx_files[0].name)

            csv = [x for x in os.scandir(path) if x.name.endswith('.csv')]
            if len(csv) == 0:
                print(f"Warning: {path}: no selected tags .csv file, skipped")
                continue

            def tag_select_csvs_up_front(k):
                k = k.name.lower()
                return -1 if "tag" in k or "select" in k else 1

            csv.sort(key=tag_select_csvs_up_front)
            tags_path = Path(path, csv[0])

            if path.name not in cls.entries:
                if path.name == 'wd-v1-4-convnextv2-tagger-v2':
                    cls.entries[path.name] = WaifuDiffusionInterrogator(
                        path.name,
                        repo_id='SmilingWolf/SW-CV-ModelZoo'
                    )
                elif path.name == 'Z3D-E621-Convnext':
                    cls.entries[path.name] = WaifuDiffusionInterrogator(
                        'Z3D-E621-Convnext')
                else:
                    raise NotImplementedError(f"Add {path.name} resolution "
                                              "similar to above here")

            cls.entries[path.name].local_model = str(local_path)
            cls.entries[path.name].local_tags = str(tags_path)

        return sorted(i.name for i in cls.entries.values())

    @staticmethod
    def load_image(path: str) -> Image:
        try:
            return Image.open(path)
        except FileNotFoundError:
            print(f'${path} not found')
        except UnidentifiedImageError:
            # just in case, user has mysterious file...
            print(f'${path} is not a  supported image type')
        except ValueError:
            print(f'${path} is not readable or StringIO')
        return None

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = None
        self.tags = None
        # run_mode 0 is dry run, 1 means run (alternating), 2 means disabled
        self.run_mode = 0 if hasattr(self, "large_batch_interrogate") else 2
        # default path if not overridden by download
        self.local_model = None
        self.local_tags = None
        # XXX don't Interrogator.refresh()-ception here

    def load(self) -> bool:
        raise NotImplementedError()

    def large_batch_interrogate(self, images: List, dry_run=False) -> str:
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if self.model is not None:
            del self.model
            self.model = None
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags
            self.tags = None

        return unloaded

    def interrogate_image(self, image: Image) -> None:
        sha = IOData.get_bytes_hash(image.tobytes())
        QData.clear(1 - Interrogator.input["cumulative"])

        fi_key = sha + self.name
        count = 0

        if fi_key in QData.query:
            # this file was already queried for this interrogator.
            QData.single_data(fi_key)
        else:
            # single process
            count += 1
            data = ('', '', fi_key) + self.interrogate(image)
            # When drag-dropping an image, the path [0] is not known
            if Interrogator.input["unload_after"]:
                self.unload()

            QData.apply_filters(data)

        for got in QData.in_db.values():
            QData.apply_filters(got)

        Interrogator.output = QData.finalize(count)

    def batch_interrogate_image(self, index: int) -> None:
        # if outputpath is '', no tags file will be written
        if len(IOData.paths[index]) == 5:
            path, out_path, output_dir, image_hash, image = IOData.paths[index]
        elif len(IOData.paths[index]) == 4:
            path, out_path, output_dir, image_hash = IOData.paths[index]
            image = Interrogator.load_image(path)
            # should work, we queried before to get the image_hash
        else:
            path, out_path, output_dir = IOData.paths[index]
            image = Interrogator.load_image(path)
            if image is None:
                return

            image_hash = IOData.get_bytes_hash(image.tobytes())
            IOData.paths[index].append(image_hash)
            if getattr(shared.opts, 'tagger_store_images', False):
                IOData.paths[index].append(image)

            if output_dir:
                output_dir.mkdir(0o755, True, True)
                # next iteration we don't need to create the directory
                IOData.paths[index][2] = ''
        QData.image_dups[image_hash].add(path)

        abspath = str(path.absolute())
        fi_key = image_hash + self.name

        if fi_key in QData.query:
            # this file was already queried for this interrogator.
            i = QData.get_index(fi_key, abspath)
            # this file was already queried and stored
            QData.in_db[i] = (abspath, out_path, '', {}, {})
        else:
            data = (abspath, out_path, fi_key) + self.interrogate(image)
            # also the tags can indicate that the image is a duplicate
            no_floats = sorted(filter(lambda x: not isinstance(x[0], float),
                                      data[3].items()), key=lambda x: x[0])
            sorted_tags = ','.join(f'({k},{v:.1f})' for (k, v) in no_floats)
            QData.image_dups[sorted_tags].add(abspath)
            QData.apply_filters(data)
            QData.had_new = True

    def batch_interrogate(self) -> None:
        """ Interrogate all images in the input list """
        QData.clear(1 - Interrogator.input["cumulative"])

        if Interrogator.input["large_query"] is True and self.run_mode < 2:
            # TODO: write specified tags files instead of simple .txt
            image_list = [str(x[0].resolve()) for x in IOData.paths]
            self.large_batch_interrogate(image_list, self.run_mode == 0)

            # alternating dry run and run modes
            self.run_mode = (self.run_mode + 1) % 2
            count = len(image_list)
            Interrogator.output = QData.finalize(count)
        else:
            verb = getattr(shared.opts, 'tagger_verbose', True)
            count = len(QData.query)

            for i in tqdm(range(len(IOData.paths)), disable=verb, desc='Tags'):
                self.batch_interrogate_image(i)

            if Interrogator.input["unload_after"]:
                self.unload()

            count = len(QData.query) - count
            Interrogator.output = QData.finalize_batch(count)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        raise NotImplementedError()


class DeepDanbooruInterrogator(Interrogator):
    """ Interrogator for DeepDanbooru models """
    def __init__(self, name: str, project_path: os.PathLike) -> None:
        super().__init__(name)
        self.project_path = project_path
        self.model = None
        self.tags = None

    def load(self) -> None:
        print(f'Loading {self.name} from {str(self.project_path)}')

        # deepdanbooru package is not include in web-sd anymore
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc663
        from launch import is_installed, run_pip
        if not is_installed('deepdanbooru'):
            package = os.environ.get(
                'DEEPDANBOORU_PACKAGE',
                'git+https://github.com/KichangKim/DeepDanbooru.'
                'git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff'
            )

            run_pip(
                f'install {package} tensorflow tensorflow-io', 'deepdanbooru')

        import tensorflow as tf

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        for device in tf.config.experimental.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError as err:
                print(err)

        with tf.device(TF_DEVICE_NAME):
            import deepdanbooru.project as ddp

            self.model = ddp.load_model_from_project(
                project_path=self.project_path,
                compile_model=False
            )

            print(f'Loaded {self.name} model from {str(self.project_path)}')

            self.tags = ddp.load_tags_from_project(
                project_path=self.project_path
            )

    def unload(self) -> bool:
        return False

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        # init model
        if self.model is None:
            if not self.load():
                return {}, {}

        import deepdanbooru.data as ddd

        # convert an image to fit the model
        image_bufs = io.BytesIO()
        image.save(image_bufs, format='PNG')
        image = ddd.load_image_for_evaluate(
            image_bufs,
            self.model.input_shape[2],
            self.model.input_shape[1]
        )

        image = image.reshape((1, *image.shape[0:3]))

        # evaluate model
        result = self.model.predict(image)

        confidences = result[0].tolist()
        ratings = {}
        tags = {}

        for i, tag in enumerate(self.tags):
            if tag[:7] != "rating:":
                tags[tag] = confidences[i]
            else:
                ratings[tag[7:]] = confidences[i]

        return ratings, tags

    def large_batch_interrogate(self, images: List, dry_run=False) -> str:
        raise NotImplementedError()


class HFInterrogator(Interrogator):
    """ Interrogator for HuggingFace models """
    def __init__(
        self,
        name: str,
        model_path: str,
        tags_path: str,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.model = None
        # tagger_hf_hub_down_opts contains args to hf_hub_download(). Parse
        # and pass only the supported args.

        signature = inspect.signature(hf_hub_download)
        self.repo_specs = {'repo_id', 'revision', 'library_name',
                           'library_version'}
        self.hf_params = {}
        for k in kwargs:
            if k in signature.parameters:
                tp = signature.parameters[k].annotation
                if isinstance(kwargs[k], tp):
                    self.hf_params[k] = kwargs[k]
                    continue
            print(f"Warning: interrogators.json: model {self.name}: "
                  f"parameter {k} unsupported or or wrong type.")

        if 'repo_id' not in self.hf_params:
            print(f"Warning: interrogators.json: HuggingFace model {self.name}"
                  " lacks a repo_id. If not already local, download may fail.")

        attrs = getattr(shared.opts, 'tagger_hf_hub_down_opts',
                        f'cache_dir="{Its.hf_cache}"')
        attrs = [attr.split('=') for attr in map(str.strip, attrs.split(','))]

        signature = inspect.signature(hf_hub_download)
        for arg, val in attrs:
            if arg == 'filename' or arg in self.repo_specs:

                print(f"Settings -> Tagger -> HuggingFace parameters: {arg}: "
                      "Specific options need to go in the interrogators.json.")

            elif arg in signature.parameters:
                try:
                    tp = signature.parameters[arg].annotation
                    self.hf_params[arg] = tp(val)

                except TypeError:
                    # unions, used for str or PathLike and a few.
                    if val == 'None':
                        self.hf_params[arg] = None
                    elif arg == 'token' and val in {'True', 'False'}:
                        self.hf_params[arg] = val == 'True'
                    else:
                        if val[0] == val[-1] and val[0] in "'\"":
                            val = val[1:-1]
                        self.hf_params[arg] = str(val)
            else:
                print(f"Settings -> Tagger -> HuggingFace parameters: {arg}: "
                      "Invalid for hf_hub_download() => ignored.")

    def download(self) -> Tuple[str, str]:
        repo_id = self.hf_params.get('repo_id', '(?)')
        print(f"Loading {self.name} model file from {repo_id}")
        if self.local_model == '':
            Interrogator.refresh()
        paths = [self.local_model, self.local_tags]

        data = {}
        for k in self.repo_specs:
            if k in self.hf_params:
                data[k] = self.hf_params[k]

        # check if the model is up to date
        info_path = Path(self.local_model).with_suffix('.info')
        if info_path.exists():

            if all(os.path.exists(p) for p in paths):
                with open(info_path, 'r') as filen:
                    try:
                        old_data = json.load(filen)
                        if old_data == data:
                            print(f"Model {self.name} is up to date.")
                            return paths
                    except json.decoder.JSONDecodeError:
                        pass

            try:
                for i, filen in enumerate([self.model_path, self.tags_path]):
                    self.hf_params['filename'] = filen
                    paths[i] = hf_hub_download(**self.hf_params)
            except Exception as err:
                print(f"hf_hub_download({self.hf_params}: {err}")
                return paths

        # write the repo_specs to a json alongside the model so we can
        # check if the model is up to date
        with open(info_path, 'w') as filen:
            json.dump(data, filen)
        return paths

    def load_model(self, model_path) -> None:
        import onnxruntime
        self.model = onnxruntime.InferenceSession(model_path,
                                                  providers=onnxrt_providers)
        print(f'Loaded {self.name} model from {model_path}')


class WaifuDiffusionInterrogator(HFInterrogator):
    """ Interrogator for Waifu Diffusion models """
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        **kwargs,
    ) -> None:
        super().__init__(name, model_path, tags_path, **kwargs)
        self.tags = None

    def update_model_json(self, model_path, tags_path):
        download_model = {
            'name': self.name,
            'model_path': model_path,
            'tags_path': tags_path,
        }
        mdir = Path(shared.models_path, 'interrogators')
        mpath = Path(mdir, 'model.json')

        data = [download_model]

        if not os.path.exists(mdir):
            os.mkdir(mdir)

        elif os.path.exists(mpath):
            with io.open(file=mpath, mode='r', encoding='utf-8') as filename:
                try:
                    data = json.load(filename)
                    # No need to append if it's already contained
                    if download_model not in data:
                        data.append(download_model)
                except json.JSONDecodeError as err:
                    print(f'Adding download_model {mpath} raised {repr(err)}')
                    data = [download_model]

        with io.open(mpath, 'w', encoding='utf-8') as filename:
            json.dump(data, filename)

    def load(self) -> bool:
        model_path, tags_path = self.download()

        if not os.path.exists(model_path):
            print(f'Model path {model_path} not found.')
            return False

        if not os.path.exists(tags_path):
            print(f'Tags path {tags_path} not found.')
            return False

        self.load_model(model_path)
        self.update_model_json(model_path, tags_path)
        self.tags = read_csv(tags_path)
        return True

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        # init model
        if self.model is None:
            if not self.load():
                return {}, {}

        # code for converting the image and running the model is taken from the
        # link below. thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = dbimutils.fill_transparent(image)

        image = asarray(image)
        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        tags = dict

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(float32)
        image = expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidences = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidences'] = confidences[0]

        # first 4 items are for rating (general, sensitive, questionable,
        # explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags

    def dry_run(self, images) -> Tuple[str, Callable[[str], None]]:

        def process_images(filepaths, _):
            lines = []
            for image_path in filepaths:
                image_path = image_path.numpy().decode("utf-8")
                lines.append(f"{image_path}\n")
            with io.open("dry_run_read.txt", "a", encoding="utf-8") as filen:
                filen.writelines(lines)

        scheduled = [f"{image_path}\n" for image_path in images]

        # Truncate the file from previous runs
        print("updating dry_run_read.txt")
        io.open("dry_run_read.txt", "w", encoding="utf-8").close()
        with io.open("dry_run_scheduled.txt", "w", encoding="utf-8") as filen:
            filen.writelines(scheduled)
        return process_images

    def run(self, images, pred_model) -> Tuple[str, Callable[[str], None]]:
        threshold = QData.threshold
        self.tags["sanitized_name"] = self.tags["name"].map(
            lambda i: i if i in Its.kaomojis else i.replace("_", " ")
        )

        def process_images(filepaths, images):
            preds = pred_model(images).numpy()

            for ipath, pred in zip(filepaths, preds):
                ipath = ipath.numpy().decode("utf-8")

                self.tags["preds"] = pred
                generic = self.tags[self.tags["category"] == 0]
                chosen = generic[generic["preds"] > threshold]
                chosen = chosen.sort_values(by="preds", ascending=False)
                tags_names = chosen["sanitized_name"]

                key = ipath.split("/")[-1].split(".")[0] + "_" + self.name
                QData.add_tags = tags_names
                QData.apply_filters((ipath, '', {}, {}), key, False)

                tags_string = ", ".join(tags_names)
                txtfile = Path(ipath).with_suffix(".txt")
                with io.open(txtfile, "w", encoding="utf-8") as filename:
                    filename.write(tags_string)
        return images, process_images

    def large_batch_interrogate(self, images, dry_run=True) -> None:
        """ Interrogate a large batch of images. """

        # init model
        if not hasattr(self, 'model') or self.model is None:
            if not self.load():
                return

        os.environ["TF_XLA_FLAGS"] = '--tf_xla_auto_jit=2 '\
                                     '--tf_xla_cpu_global_jit'
        # Reduce logging
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        import tensorflow as tf

        from tagger.generator.tf_data_reader import DataGenerator

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for device in gpus:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except RuntimeError as err:
                    print(err)

        if dry_run:  # dry run
            height, width = 224, 224
            process_images = self.dry_run(images)
        else:
            _, height, width, _ = self.model.inputs[0].shape

            @tf.function
            def pred_model(model):
                return self.model(model, training=False)

            process_images = self.run(images, pred_model)

        generator = DataGenerator(
            file_list=images, target_height=height, target_width=width,
            batch_size=getattr(shared.opts, 'tagger_batch_size', 1024)
        ).gen_ds()

        orig_add_tags = QData.add_tags
        for filepaths, image_list in tqdm(generator):
            process_images(filepaths, image_list)
        QData.add_tag = orig_add_tags
        del os.environ["TF_XLA_FLAGS"]


class MLDanbooruInterrogator(HFInterrogator):
    """ Interrogator for the MLDanbooru model. """
    def __init__(
        self,
        name: str,
        model_path: str,
        tags_path='classes.json',
        **kwargs
    ) -> None:
        super().__init__(name, model_path, tags_path, **kwargs)
        self.tags = None

    def load(self) -> bool:
        model_path, tags_path = self.download()

        if not os.path.exists(model_path):
            print(f'Model path {model_path} not found.')
            return False

        if not os.path.exists(tags_path):
            print(f'Tags path {tags_path} not found.')
            return False

        self.load_model(model_path)

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.tags = json.load(filen)

        return True

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            if not self.load():
                return {}, {}

        image = dbimutils.fill_transparent(image)
        image = dbimutils.resize(image, 448)  # TODO CUSTOMIZE

        x = asarray(image, dtype=float32) / 255
        # HWC -> 1CHW
        x = x.transpose((2, 0, 1))
        x = expand_dims(x, 0)

        input_ = self.model.get_inputs()[0]
        output = self.model.get_outputs()[0]
        # evaluate model
        y, = self.model.run([output.name], {input_.name: x})

        # Softmax
        y = 1 / (1 + exp(-y))

        tags = {tag: float(conf) for tag, conf in zip(self.tags, y.flatten())}
        return {}, tags

    def large_batch_interrogate(self, images: List, dry_run=False) -> str:
        raise NotImplementedError()
