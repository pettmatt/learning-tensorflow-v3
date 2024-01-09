# Converting models

Keras or HDF5 (Hierarchical data format v5) models can be converted to be used in TensorFlow.js.
Because these models are used in Tensorflow, we need to convert models using python tools in order to use them in TensorFlow.js API environment.

At the time of writing the converter wizard tool has some issues, which should be fixed in near future, when new version of tensorflow is released (just my luck). We shall see if that is going to happen https://github.com/google/jax/issues/18978.