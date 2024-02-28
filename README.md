# Neural Certificates

A prototype learner-verifier for dynamical systems. 

# Cautions for using this repository

Most of the code here is written experimentally.
Some modules are stale, and will not work.
Some modules are duplicates of other modules.
My _suggestion_ is to look at the drafts;
at commit 
[a1830c1](https://github.com/mahykari/neural-certificates/tree/a1830c1f75d4e77e7f987b30f18aa73c801c96e1),
all drafts work.

# Using Maraboupy

We use the Marabou verifier via its Python interface.
To build Marabou (and _Maraboupy_),
follow the instructions on 
the Marabou installation page
at [this link](https://neuralnetworkverification.github.io/Marabou/Setup/0_Installation.html). 
**Before** compiling,
make sure to change 
the parameter `INTERVAL_SPLITTING_FREQUENCY`
(in [this file](https://github.com/NeuralNetworkVerification/Marabou/blob/8129640537d63deac485daaf0f2f1c09e247e928/src/configuration/GlobalConfiguration.cpp#L63))
to `1`.
This might cause some tests to fail; it is OK.

After building Marabou, make a link to the `maraboupy`
directory, like the following:
```bash
ln -s /path/to/Marabou/maraboupy .
```
This lets us use `maraboupy` as a module in this project.

# Using `neural_clbf`

The project `neural_clbf` is available at 
[this page](https://github.com/MIT-REALM/neural_clbf.git).
Follow the instructions on their page.
You might want to make a link to `neural_clbf`.
Steps are the same as the case with `maraboupy`.

# The `playground` notebook

As the name suggests, it is only a playground.
All useful things from the playground are 
already implemented in the source files (or drafts).