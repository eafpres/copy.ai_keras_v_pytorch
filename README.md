# copy.ai_keras_v_pytorch

This repo contains results of asking Chat from copy.ai two questions:

Can you show me Python code using Keras for a basic CNN model?
Can you show me the same model, but in PyTorch?

The raw output of this dialogue is in the file keras_v_pytorch_raw.py

The code was reasonable, but had various issues.
For instance, in the keras code, the imports were not quite correct for TF 2.0.
In the PyTorch code, a closing parenthesis was missing, and one comment was mis-aligned.

In both cases, there was nothing provided to load the data, just an explanation about doing that.
The final code to compare the basic model on an existing dataset is in the code folder as keras_v_pytorch.py
Data loaders and other needed things have been added.  In particular, the CNN model itself was 
simplified mainly to speed up training as this is only a demonstration.

In the PyTorch code, I wanted to monitor the accuracy not just the loss, so that section of code
was changed and tqdm added to format the output a bit.

In the end, both models worked similarly, and were decent starting points, but the code in either
case would not have run as-is.
