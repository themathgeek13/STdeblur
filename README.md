# STdeblur
Spatiotemporal deblurring in PyTorch

* Adversarial Spatio-Temporal Learning for Video Deblurring: https://arxiv.org/pdf/1804.00533.pdf

The architecture has been implemented (with shuffling, 3D convolutions and 1x1 convolutions, as well as with ResNet blocks) in PyTorch. The dataset being used is a modified version of the Deep Deblur dataset (https://github.com/SeungjunNah/DeepDeblur_release), which can be found here: https://github.com/KupynOrest/DeblurGAN.

Currently it uses a batch-size of 1, and SGD with momentum. This can easily be modified.

NOTE: The implementation is complete, however I was unable to get it to work despite following the above paper implementation.
