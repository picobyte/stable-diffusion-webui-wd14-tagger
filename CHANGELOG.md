# v1.1.2 (2023-08-26)

Explain recursive path usage better in ui
Fix sending tags via buttons to txt2img and img2img
type additions, inadvertently pushed, later retouched.
allow setting gpu device via flag
Fix inverted cumulative checkbox
wrap_gradio_gpu_call fallback
Fix for preload shared access
preload update
A few ui changes
Fix not clearing the tags after writing them to files
Fix: Tags were still added, beyond count threshold
fix search/replace bug
(here int based weights were reverted)
circumvent when unable to load tensorflow
fix for too many exclude_tags
add db.json validation schema, add schema validation
return fix for fastapi
pick up huggingface cache dir from env, with default, configurable also via settings.
leave tensorflow requirements to the user.
Fix for Reappearance of gradio bug: duplicate image edit
(index based weights, but later reverted)
Instead of cache_dir use local_dir, leav


# v1.1.1 eada050 (2023-07-20) 

Internal cleanup, no separate interrogation for inverse
Fix issues with search and sending selection to keep/exclude
Fix issue #14, picking up last edit box changes
Fix 2 issues reported by guansss
fix huggingface reload issues. Thanks to Atoli and coder168 for reporting
experimental tensorflow unloading, but after some discussion, maybe conversion to onxx can solve this. See #17, thanks again Sean Wang.
add gallery tab, rudimentary.
fix some hf download issues
fixes for fastapi
added ML-Danbooru support, thanks to [CCRcmcpe](github.com/CCRcmcpe)


# v1.1.0 87706b7 (2023-07-16)

fix: failed to install onnxruntime package on MacOS thanks to heady713
fastapi: remote unload model, picked up from [here](https://github.com/toriato/stable-diffusion-webui-wd14-tagger/pull/109)
attribute error fix from aria1th also reported by yjunej
re-allowed weighted tags files, now configured in settings -> tagger.
wzgrx pointed out there were some modules not installed by default, so I've added a requirements.txt file that will auto-install required dependencies. However, the initial requirements.txt had issues. I ran to create the requirements.txt:
```
pipreqs --force `pwd`
sed -i s/==.*$//g requirements.txt
```
but it ended up adding external modules that were shadowing webui modules. If you have installed those, you may find you are not even able to start the webui until you remove them. Change to the directory of my extension and
```
pip uninstall webui
pip uninstall modules
pip uninstall launch
```
In particular installing a module named modules was a serious problem. Python should flag that name as illegal.

There were some interrogators that were not working unless you have them installed manually. Now they are only listed if you have them.

Thanks to wzgrx for testing and reporting these last two issues.
changed internal file structure, thanks to idiotcomerce #4
more regex usage in search and exclusion tags
fixed a bug where some exclusion tags were not reflected in the tags file
changed internal error handling, It is a bit quirky, which I intend to fix, still.
If you find it keeps complaining about an input field without reason, just try editing that one again (e.g. add a space there and remove it).


# v1.0.0 a1b59d6 (2023-07-10)

You may have to remove the presets/default.json and save a new one.witth your desired defaults. Otherwise checkboxes may not have the right default values.

General changes:

Weights, when enabled, are not printed in the tags list. Weights are displayed in the list below already as bars, so they do not add information, only obfuscate the list IMO.
There is an settings entry for the tagger, several options have been moved there.
The list of tags weights stops at a number specified on the settings tab (the slider)
There is both an included and excluded rags tab
tags in the tags list on top are clickable.
Tags below are also clickable. There is a difference if you click on the dotted line or on the actual word. a click on the word will add it to a search/kept tag (dependent on which was last active) on the dotted line will add it to the input box next to it.
interrogations can be combined (checkbox), also for a single image.
Make the labels listed clickable again, a click will add it to the selected listbox. This also functions when you are on the discarded tags tab.
Added search and replace input lists.
Changed behavior: when clicking on the dotted line, inserted is in the exclude/replace input list, if not the tag is inserted in the additional/search input list
Added a Mininmum fraction for tags slider. This filters tags based on the fraction of images and interrogations per image that has this tag with the selected weight threshold. I find this kind of filtering makes more sense than limiting the tags list to a number, though that is ok to prevent cluttering up the view,

Added a string search selected tags input field (top right) and two buttons:
Move visible tags to keep tags
Move visible tags to exclude tags

For batch processing:
After each update a db.json is written in the images folder. The db contains the weights for queries, a rerun of the same images using an interrogator just rereads this db.json. This also works after a stable diffusion reload or a reboot, as long as this db.json is there.

There is a huge batch implementation, but I was unable to test, not the right tensorflow version. EXPERIMENTAL. It is only enabled if you have the right tf version, but it's likely buggy due to my lack of testing. feel free to send me a patch if you can improve it. also see here
pre- or appending weights to weighed tag files, i.e. with weights enabled, will instead have the weights averaged

After batch processing the combined tag count average is listed, for all processed files, and the corrected average when combining the weighed tags. This is not limited to the tag_count_threshold, as it relates to the weights of all tag files. Conversely, the already existing threshold slider does affect this list length.
search tag can be a single regex or as many as replacements, comma separated. Currently a single regex or multiple as many strings in search an replace are allowed, but this is going to change in the near future, to allow all regexes and back referencing per replacements as in a re.sub().
added a 'verbose setting'.
a comma was previously missing when appending tags
several of the interrogators have been fixed.



