# Active inference agent

This script implements a simple active inference agent following the description in "A Concise Mathematical Description of Active Inference in Discrete Time", by Jesse van Oostrum, Carlotta Langer and Nihat Ay ([arXiv:2406.07726](https://arxiv.org/abs/2406.07726)).
In the comments of the methods in the `agent` class we refer to the corresponding equations in the paper. 

An example usage can be found in the file `tmaze.py`.

The most important thing to note is that in this implementation all observation modalities and state factors are flattened into one single dimension. This significantly simplifies the implementation of the main equations of active inference, compared to other implementations such as `SPM` and `pymdp`. We use the class Flattener to translate between the original (unflattened) observations and states and their flattened versions. 

In `tmaze-learning.py` you can find an example of how the agent would update the A and B matrices. However, this environment is not rich enough for these matrices to be fully learned. 

If you use this code in your work or research, please consider citing our paper:
```
@article{van2024concise,
  title={A Concise Mathematical Description of Active Inference in Discrete Time},
  author={van Oostrum, Jesse and Langer, Carlotta and Ay, Nihat},
  journal={arXiv preprint arXiv:2406.07726},
  year={2024}
}
```

