to build training data
python load_data.py directory_of_raw_pngs directory_of_gt_pngs

to train
python unet.py

Code does not currently save network or weights, I usually do this:
python -i unet.py
<<< wait some hours >>>
Control-C
>>> model.save_weights('weights.h5')
>>> open("unetv2.json", "w").write(model.to_json())
