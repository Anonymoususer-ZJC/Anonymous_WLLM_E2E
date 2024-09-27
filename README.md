# An End-to-End Model for Logits Based Large Language Models Watermarking

This is an anonymous repository for our paper An End-to-End Model for Logits Based Large Language Models Watermarking submitted on ICLR 2025. During the paper review period, we would like to use this temporary anonymous repository for code release.

### Repository Structure
The project structure is as follows.
- `main.py` is the main entry point for the code.
- `train_e2e.py` is the train script for the e2e model.
- `Dataset/` stores the training dataset.
- `WLLM/` stores the main model weights we used in our evaluation experiment (for your convinence, you can direxctly use this model for evalution).
- `evaluation/` contains the rest of the evaluation code, namely:
  -`config/`
  -`watermark/`
  -`XXX/`
  
### Basic Setup

#### Setting up the environment

- python 3.9
- pytorch
- pip install -r requirements.txt

*Tips:* If you wish to utilize the EXPEdit or ITSEdit algorithm in the evaluation step, you will need to import for .pyx file, take EXPEdit as an example:

- run `python watermark/exp_edit/cython_files/setup.py build_ext --inplace`
- move the generated `.so` file into `watermark/exp_edit/cython_files/`

### Runing the Code

### Contact
Once the double-blind review is complete, we will update our code repository and publish the authors’ contact information. Thank you very much for your interest.
