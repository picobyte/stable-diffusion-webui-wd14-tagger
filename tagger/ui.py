""" This module contains the ui for the tagger tab. """
from typing import Dict, Tuple, List, Optional
import gradio as gr
import re
from PIL import Image
from packaging import version

try:
    from tensorflow import __version__ as tf_version
except ImportError:
    tf_version = '0.0.0'

from html import escape as html_esc

from modules import ui  # pylint: disable=import-error
from modules import generation_parameters_copypaste as parameters_copypaste  # pylint: disable=import-error # noqa

try:
    from modules.call_queue import wrap_gradio_gpu_call
except ImportError:
    from webui import wrap_gradio_gpu_call  # pylint: disable=import-error
from tagger import utils  # pylint: disable=import-error
from tagger.interrogator import Interrogator as It  # pylint: disable=E0401
from tagger.uiset import IOData, QData  # pylint: disable=import-error

TAG_INPUTS = ["add", "keep", "exclude", "search", "replace"]
COMMON_OUTPUT = Tuple[
    Optional[str],               # tags as string
    Optional[str],               # html tags as string
    Optional[str],               # discarded tags as string
    Optional[Dict[str, float]],  # rating confidences
    Optional[Dict[str, float]],  # tag confidences
    Optional[Dict[str, float]],  # excluded tag confidences
    str,               # error message
]


def unload_interrogators() -> Tuple[str]:
    unloaded_models = 0
    remaining_models = ''

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1
        elif i.model is not None:
            if remaining_models == '':
                remaining_models = f', remaining models:<ul><li>{i.name}</li>'
            else:
                remaining_models = remaining_models + f'<li>{i.name}</li>'
    if remaining_models != '':
        remaining_models = remaining_models + "Some tensorflow models could "\
                           "not be unloaded, a known issue."
    QData.clear(1)

    return (f'{unloaded_models} model(s) unloaded{remaining_models}',)


def on_interrogate(
    input_glob: str, output_dir: str, name: str, filt: str, *args
) -> COMMON_OUTPUT:
    # input glob should always be rechecked for new files
    IOData.update_input_glob(input_glob)
    if output_dir != It.input["output_dir"]:
        IOData.update_output_dir(output_dir)
        It.input["output_dir"] = output_dir

    if len(IOData.err) > 0:
        return (None,) * 6 + (IOData.error_msg(),)

    for i, val in enumerate(args):
        part = TAG_INPUTS[i]
        if val != It.input[part]:
            getattr(QData, "update_" + part)(val)
            It.input[part] = val

    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None,) * 6 + (f"'{name}': invalid interrogator",)

    interrogator.batch_interrogate()
    return search_filter(filt)


def on_gallery() -> List:
    return QData.get_image_dups()


def on_interrogate_image(*args) -> COMMON_OUTPUT:
    # hack brcause image interrogaion occurs twice
    It.odd_increment = It.odd_increment + 1
    if It.odd_increment & 1 == 1:
        return (None,) * 6 + ('',)
    return on_interrogate_image_submit(*args)


def on_interrogate_image_submit(
    image: Image, name: str, filt: str, *args
) -> COMMON_OUTPUT:
    for i, val in enumerate(args):
        part = TAG_INPUTS[i]
        if val != It.input[part]:
            getattr(QData, "update_" + part)(val)
            It.input[part] = val

    if image is None:
        return (None,) * 6 + ('No image selected',)
    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None,) * 6 + (f"'{name}': invalid interrogator",)

    interrogator.interrogate_image(image)
    return search_filter(filt)


def move_selection_to_input(
    filt: str, field: str
) -> Tuple[Optional[str], Optional[str], str]:
    """ moves the selected to the input field """
    if It.output is None:
        return (None, None, '')
    tags = It.output[1]
    got = It.input[field]
    existing = set(got.split(', '))
    if filt:
        re_part = re.compile('(' + re.sub(', ?', '|', filt) + ')')
        tags = {k: v for k, v in tags.items() if re_part.search(k) and
                k not in existing}
        print("Tags remaining: ", tags)

    if len(tags) == 0:
        return ('', None, '')

    if got != '':
        got = got + ', '

    (data, info) = It.set(field)(got + ', '.join(tags.keys()))
    return ('', data, info)


def move_selection_to_keep(
    tag_search_filter: str
) -> Tuple[Optional[str], Optional[str], str]:
    return move_selection_to_input(tag_search_filter, "keep")


def move_selection_to_exclude(
    tag_search_filter: str
) -> Tuple[Optional[str], Optional[str], str]:
    return move_selection_to_input(tag_search_filter, "exclude")


def search_filter(filt: str) -> COMMON_OUTPUT:
    """ filters the tags and lost tags for the search field """
    ratings, tags, lost, info = It.output
    if ratings is None:
        return (None,) * 6 + (info,)
    if filt:
        re_part = re.compile('(' + re.sub(', ?', '|', filt) + ')')
        tags = {k: v for k, v in tags.items() if re_part.search(k)}
        lost = {k: v for k, v in lost.items() if re_part.search(k)}

    h_tags = ', '.join(f'<a href="javascript:tag_clicked(\'{html_esc(k)}\','
                       f'true)">{k}</a>' for k in tags.keys())
    h_lost = ', '.join(f'<a href="javascript:tag_clicked(\'{html_esc(k)}\','
                       f'false)">{k}</a>' for k in lost.keys())

    return (', '.join(tags.keys()), h_tags, h_lost, ratings, tags, lost, info)


def on_ui_tabs():
    """ configures the ui on the tagger tab """
    # If checkboxes misbehave you have to adapt the default.json preset
    tag_input = {}

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row():
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )
                        image_submit = gr.Button(
                            value='Interrogate image',
                            variant='primary'
                        )

                    with gr.TabItem(label='Batch from directory'):
                        input_glob = utils.preset.component(
                            gr.Textbox,
                            value='',
                            label='Input directory - To recurse use ** or */* '
                                  'in your glob; also check the settings tab.',
                            placeholder='/path/to/images or to/images/**/*'
                        )
                        output_dir = utils.preset.component(
                            gr.Textbox,
                            value=It.input["output_dir"],
                            label='Output directory',
                            placeholder='Leave blank to save images '
                                        'to the same path.'
                        )

                        batch_submit = gr.Button(
                            value='Interrogate',
                            variant='primary'
                        )
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='panel'):
                                large_query = utils.preset.component(
                                    gr.Checkbox,
                                    label='huge batch query (TF 2.10, '
                                    'experimental)',
                                    value=False,
                                    interactive=version.parse(tf_version) ==
                                    version.parse('2.10')
                                )
                            with gr.Column(variant='panel'):
                                save_tags = utils.preset.component(
                                    gr.Checkbox,
                                    label='Save to tags files',
                                    value=True
                                )

                info = gr.HTML(
                    label='Info',
                    interactive=False,
                    elem_classes=['info']
                )

                # interrogator selector
                with gr.Column():
                    # preset selector
                    with gr.Row(variant='compact'):
                        available_presets = utils.preset.list()
                        selected_preset = gr.Dropdown(
                            label='Preset',
                            choices=available_presets,
                            value=available_presets[0]
                        )

                        save_preset_button = gr.Button(
                            value=ui.save_style_symbol
                        )

                        ui.create_refresh_button(
                            selected_preset,
                            lambda: None,
                            lambda: {'choices': utils.preset.list()},
                            'refresh_preset'
                        )

                    with gr.Row(variant='compact'):
                        def refresh():
                            utils.refresh_interrogators()
                            return sorted(x.name for x in utils.interrogators
                                                               .values())
                        interrogator_names = refresh()
                        interrogator = utils.preset.component(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )

                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': refresh()},
                            'refresh_interrogator'
                        )

                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )
                    with gr.Row(variant='compact'):
                        tag_input["add"] = utils.preset.component(
                            gr.Textbox,
                            label='Additional tags (comma split)',
                            elem_id='additional-tags'
                        )
                    with gr.Row(variant='compact'):
                        threshold = utils.preset.component(
                            gr.Slider,
                            label='Weight threshold',
                            minimum=0,
                            maximum=1,
                            value=QData.threshold
                        )
                        tag_frac_threshold = utils.preset.component(
                            gr.Slider,
                            label='Min tag fraction in batch and '
                                  'interrogations',
                            minimum=0,
                            maximum=1,
                            value=QData.tag_frac_threshold,
                        )
                    with gr.Row(variant='compact'):
                        cumulative = utils.preset.component(
                            gr.Checkbox,
                            label='Combine interrogations',
                            value=False
                        )
                        unload_after = utils.preset.component(
                            gr.Checkbox,
                            label='Unload model after running',
                            value=False
                        )
                    with gr.Row(variant='compact'):
                        tag_input["search"] = utils.preset.component(
                            gr.Textbox,
                            label='Search tag, .. ->',
                            elem_id='search-tags'
                        )
                        tag_input["replace"] = utils.preset.component(
                            gr.Textbox,
                            label='-> Replace tag, ..',
                            elem_id='replace-tags'
                        )
                    with gr.Row(variant='compact'):
                        tag_input["keep"] = utils.preset.component(
                            gr.Textbox,
                            label='Keep tag, ..',
                            elem_id='keep-tags'
                        )
                        tag_input["exclude"] = utils.preset.component(
                            gr.Textbox,
                            label='Exclude tag, ..',
                            elem_id='exclude-tags'
                        )

            # output components
            with gr.Column(variant='panel'):
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        mv_selection_to_keep = gr.Button(
                            value='Move visible tags to keep tags',
                            variant='secondary'
                        )
                        mv_selection_to_exclude = gr.Button(
                            value='Move visible tags to exclude tags',
                            variant='secondary'
                        )
                    with gr.Column(variant='compact'):
                        tag_search_selection = utils.preset.component(
                            gr.Textbox,
                            label='Multi string search: part1, part2.. '
                                  '(Enter key to update)',
                        )
                with gr.Tabs():
                    with gr.TabItem(label='Ratings and included tags'):
                        # clickable tags to populate excluded tags
                        tags = gr.State(value="")
                        html_tags = gr.HTML(
                            label='Tags',
                            elem_id='tags',
                        )

                        with gr.Row():
                            parameters_copypaste.bind_buttons(
                                parameters_copypaste.create_buttons(
                                    ["txt2img", "img2img"],
                                ),
                                None,
                                tags
                            )
                        rating_confidences = gr.Label(
                            label='Rating confidences',
                            elem_id='rating-confidences',
                        )
                        tag_confidences = gr.Label(
                            label='Tag confidences',
                            elem_id='tag-confidences',
                        )
                    with gr.TabItem(label='Excluded tags'):
                        # clickable tags to populate keep tags
                        discarded_tags = gr.HTML(
                            label='Tags',
                            elem_id='tags',
                        )
                        excluded_tag_confidences = gr.Label(
                            label='Excluded Tag confidences',
                            elem_id='discard-tag-confidences',
                        )
                    tab_gallery = gr.TabItem(label='Gallery')
                    with tab_gallery:
                        gallery = gr.Gallery(
                            label='Gallery',
                            elem_id='gallery',
                            columns=[2],
                            rows=[8],
                            object_fit="contain",
                            height="auto"
                        )

        # register events
        # Checkboxes
        cumulative.input(fn=It.flip('cumulative'), inputs=[], outputs=[])
        large_query.input(fn=It.flip('large_query'), inputs=[], outputs=[])
        unload_after.input(fn=It.flip('unload_after'), inputs=[], outputs=[])

        save_tags.input(fn=IOData.flip_save_tags(), inputs=[], outputs=[])

        # Preset and unload buttons
        selected_preset.change(fn=utils.preset.apply, inputs=[selected_preset],
                               outputs=[*utils.preset.components, info])

        save_preset_button.click(fn=utils.preset.save, inputs=[selected_preset,
                                 *utils.preset.components], outputs=[info])

        unload_all_models.click(fn=unload_interrogators, outputs=[info])

        # Sliders
        threshold.input(fn=QData.set("threshold"), inputs=[threshold],
                        outputs=[])
        threshold.release(fn=QData.set("threshold"), inputs=[threshold],
                          outputs=[])

        tag_frac_threshold.input(fn=QData.set("tag_frac_threshold"),
                                 inputs=[tag_frac_threshold], outputs=[])
        tag_frac_threshold.release(fn=QData.set("tag_frac_threshold"),
                                   inputs=[tag_frac_threshold], outputs=[])

        # Input textboxes (blur == lose focus)
        for tag in TAG_INPUTS:
            tag_input[tag].blur(fn=wrap_gradio_gpu_call(It.set(tag)),
                                inputs=[tag_input[tag]],
                                outputs=[tag_input[tag], info])

        input_glob.blur(fn=wrap_gradio_gpu_call(It.set("input_glob")),
                        inputs=[input_glob], outputs=[input_glob, info])
        output_dir.blur(fn=wrap_gradio_gpu_call(It.set("output_dir")),
                        inputs=[output_dir], outputs=[output_dir, info])

        tab_gallery.select(fn=on_gallery, inputs=[], outputs=[gallery])

        common_output = [tags, html_tags, discarded_tags, rating_confidences,
                         tag_confidences, excluded_tag_confidences, info]

        # search input textbox
        for fun in [tag_search_selection.change, tag_search_selection.submit]:
            fun(fn=wrap_gradio_gpu_call(search_filter),
                inputs=[tag_search_selection], outputs=common_output)

        # buttons to move tags (right)
        mv_selection_to_keep.click(
            fn=wrap_gradio_gpu_call(move_selection_to_keep),
            inputs=[tag_search_selection],
            outputs=[tag_search_selection, tag_input["keep"], info])

        mv_selection_to_exclude.click(
            fn=wrap_gradio_gpu_call(move_selection_to_exclude),
            inputs=[tag_search_selection],
            outputs=[tag_search_selection, tag_input["exclude"], info])

        common_input = [interrogator, tag_search_selection] + \
                       [tag_input[tag] for tag in TAG_INPUTS]

        # interrogation events
        image_submit.click(fn=wrap_gradio_gpu_call(on_interrogate_image_submit),
             inputs=[image] + common_input, outputs=common_output)

        image.change(fn=wrap_gradio_gpu_call(on_interrogate_image),
             inputs=[image] + common_input, outputs=common_output)

        batch_submit.click(fn=wrap_gradio_gpu_call(on_interrogate),
                           inputs=[input_glob, output_dir] + common_input,
                           outputs=common_output)

    return [(tagger_interface, "Tagger", "tagger")]
