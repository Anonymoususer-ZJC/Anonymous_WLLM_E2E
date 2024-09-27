# An End-to-End Model for Logits Based Large Language Models Watermarking

This is an anonymous repository for our paper An End-to-End Model for Logits Based Large Language Models Watermarking submitted on ICLR2025. During the paper review period, we would like to use this temporary anonymous repository for code release.

### Basic Setup

#### Setting up the environment

- python 3.9
- pytorch
- pip install -r requirements.txt

*Tips:* If you wish to utilize the EXPEdit or ITSEdit algorithm in the evaluation step, you will need to import for .pyx file, take EXPEdit as an example:

- run `python watermark/exp_edit/cython_files/setup.py build_ext --inplace`
- move the generated `.so` file into `watermark/exp_edit/cython_files/`


#### Applying evaluation pipelines
```

### More user examples

Additional user examples are available in `test/`. To execute the scripts contained within, first run the following command to set the environment variables.

```bash
export PYTHONPATH="path_to_the_MarkLLM_project:$PYTHONPATH"
```
### Repository Structure
The project structure is as follows.
- XXXXXX
- XXXXXX
- XXXXXX

### Contact
Once the double-blind review is complete, we will update our code repository and publish the authors’ contact information. Thank you very much for your interest.
