"""Settings tab entries for the tagger module"""
import os
from typing import List
from modules import shared  # pylint: disable=import-error
from gradio import inputs as gr
from huggingface_hub import hf_hub_download

# kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
DEFAULT_KAMOJIS = '0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'  # pylint: disable=line-too-long # noqa: E501

DEFAULT_OFF = '[name].[output_extension]'

HF_CACHE = os.environ.get(
    'HUGGINGFACE_HUB_CACHE',  # defaults to "$HF_HOME/hub"
    str(os.path.join(shared.models_path, 'interrogators')))


def slider_wrapper(value, elem_id, **kwargs):
    # required or else gradio will throw errors
    return gr.Slider(**kwargs)


def on_ui_settings():
    """Called when the UI settings tab is opened"""
    Its = InterrogatorSettings
    section = 'tagger', 'Tagger'
    shared.opts.add_option(
        key='tagger_out_filename_fmt',
        info=shared.OptionInfo(
            DEFAULT_OFF,
            label='Tag file output format. Leave blank to use same filename or'
            ' e.g. "[name].[hash:sha1].[output_extension]". Also allowed are '
            '[extension] or any other [hash:<algorithm>] supported by hashlib',
            section=section,
        ),
    )
    shared.opts.onchange(
        key='tagger_out_filename_fmt',
        func=Its.set_output_filename_format
    )
    shared.opts.add_option(
        key='tagger_count_threshold',
        info=shared.OptionInfo(
            100.0,
            label="Maximum number of tags to be shown in the UI",
            section=section,
            component=slider_wrapper,
            component_args={"minimum": 1.0, "maximum": 500.0, "step": 1.0},
        ),
    )
    shared.opts.add_option(
        key='tagger_batch_recursive',
        info=shared.OptionInfo(
            True,
            label='Glob recursively with input directory pattern',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_auto_serde_json',
        info=shared.OptionInfo(
            True,
            label='Auto load and save JSON database',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_store_images',
        info=shared.OptionInfo(
            False,
            label='Store images in database',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_weighted_tags_files',
        info=shared.OptionInfo(
            False,
            label='Write weights to tags files',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_verbose',
        info=shared.OptionInfo(
            False,
            label='Console log tag counts per file, no progress bar',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_repl_us',
        info=shared.OptionInfo(
            True,
            label='Use spaces instead of underscore',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_repl_us_excl',
        info=shared.OptionInfo(
            DEFAULT_KAMOJIS,
            label='Excudes (split by comma)',
            section=section,
        ),
    )
    shared.opts.onchange(
        key='tagger_repl_us_excl',
        func=Its.set_us_excl
    )
    shared.opts.add_option(
        key='tagger_escape',
        info=shared.OptionInfo(
            False,
            label='Escape brackets',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_batch_size',
        info=shared.OptionInfo(
            1024,
            label='batch size for large queries',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_hf_hub_down_opts',
        info=shared.OptionInfo(
            str(f'cache_dir="{HF_CACHE}"'),
            label='HuggingFace parameters, Comma delimited: arg=value, '
            'see huggingface_hub docs for available or leave alone.',
            section=section,
        ),
    )


def split_str(string: str, separator=',') -> List[str]:
    return [x.strip() for x in string.split(separator) if x]


class InterrogatorSettings:
    kamojis = set(split_str(DEFAULT_KAMOJIS))
    output_filename_format = DEFAULT_OFF
    hf_cache = HF_CACHE

    @classmethod
    def set_us_excl(cls):
        ruxs = getattr(shared.opts, 'tagger_repl_us_excl', DEFAULT_KAMOJIS)
        cls.kamojis = set(split_str(ruxs))

    @classmethod
    def set_output_filename_format(cls):
        fnfmt = getattr(shared.opts, 'tagger_out_filename_fmt', DEFAULT_OFF)
        if fnfmt[-12:] == '.[extension]':
            print("refused to write an image extension")
            fnfmt = fnfmt[:-12] + '.[output_extension]'

        cls.output_filename_format = fnfmt.strip()
