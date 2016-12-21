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


Data for training is stored in hd5 files in the training_data subdirectory.   Each hdf5 should have a 'raw' and 'gt' volume.  The raw volume should be 0-255 (image intensity) and 3-dimensional with Z first.  The gt volume is 3 channel, with shape ZXYC.  C=0 is the membrane channel, C=1,2 is pre/post synaptic.  0.5 in any GT channel is a "don't care" value.
