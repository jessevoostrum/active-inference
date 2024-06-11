# Active inference agent

This script implements a simple active inference agent following the description in "A concise mathematical description of active inference in discrete time", by Jesse van Oostrum, Carlotta Langer and Nihat Ay. 

An example usage can be found in the file `tmaze.py`. 

The most important thing to note is that in this implementation all observation modalities and state factors are flattened into one single dimension, which significantly simplifies the main equations of active inference. We use the class Flattener to translate between the orignal (unflattened) observations and states and their flattened versions. 