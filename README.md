#### Installation

    git clone git@github.com:ahmetustun/t5x-demo.git
    cd t5x-demo

    python3 -m pip install -e '.[tpu]' -f \
      https://storage.googleapis.com/jax-releases/libtpu_releases.html

#### Example scripts

Fine-tuning T5 for sst2

    bash scripts/t5-sst2-demo.sh gs://your-bucket/path/to/model_dir gs://your-bucket/path/to/tfds/cache"

Fine-tuning mT5 xnli

    bash scripts/mt5-xnli-demo.sh gs://your-bucket/path/to/model_dir gs://your-bucket/path/to/tfds/cache"

#### Acknowledgement
This repository is prepared for demonstration purposes. Dataset mixtures and scripts are taken or adapted from [Prompt Tuning](https://github.com/google-research/prompt-tuning) and [Multilingual T5](https://github.com/google-research/multilingual-t5) repositories.  
