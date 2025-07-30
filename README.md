# Savanna finetuning guide

This repo contains instructions (in this README) and scripts, configs, etc, for
finetuning Evo 2 7B using the [savanna](https://github.com/Zymrael/savanna)
repository. I haven't tried finetuning Evo 2 1B or 40B using savanna, but they
*should* work similarly in principle; you can build off of these scripts.

* Process your dataset from JSONL (*must* be JSONL; TXT doesn't work reliably)
  to BIN and IDX files using [this
  script](https://github.com/Zymrael/savanna/blob/main/tools/preprocess_data.py).
  For the tokenizer type, use CharLevelTokenizer. You can run the script from
  the terminal using a command like the one below. Notice that without the
  `--enforce-sample-length` flag, you get [contiguous
  packing](https://huggingface.co/blog/sirluk/llm-sequence-packing#the-solution-sequence-packing),
  whereas with the flag, you get right-padding. You might want one or the other
  depending on your use case.

```bash
python tools/preprocess_data.py --input data/raw/data_for_train.jsonl --output-prefix data/run/padded4096_data/train --workers 8 --enforce-sample-length 4096 --tokenizer-type CharLevelTokenizer
```

* This repo contains the model YAML config and Slurm script for finetuning Evo 2
  7B.
* Download [this HF model](https://huggingface.co/arcinstitute/savanna_evo2_7b)
  as your base checkpoint to finetune. Place it in a directory called
  `savanna_evo2_7b` (change this name if you like).
* Inside `savanna_evo2_7b`, place the PT file you download from HF in a
  *sub*directory called `global_step1` (the number after `global_step` doesn't
  matter; just needs to be positive). Rename the PT file itself to
  `mp_rank_00_model_states.pt`. This step is crucial for DeepSpeed to work
  properly.
* After renaming the PT file and moving it to the subdirectory `global_step1`,
  go to the model YAML config and edit the "load" field. Enter `savanna_evo2_7b`
  (i.e., the name of the directory that contains the subdirectory
  `global_step1`).
* The HF checkpoint you downloaded has a context window of 1 million. CUDA will
  almost certainly go out of memory without parallelism for this context length.
  Moreover, you probably don't need this context window for most finetuning
  tasks. Use the script `extend_filter.py` in this repo to edit the checkpoint.
  The flag `--target_seq_len` should be the context window you plan to use.
  Usage is

```bash
python extend_filter.py \
      --source_dir      ckpt/evo2_7b/global_step1 \
      --output_dir      ckpt/evo2_7b_4k_context_length/global_step1 \
      --num_groups      256 \
      --seq_len         1048576 \
      --target_seq_len  4096 \
      --overwrite
```

* Clone [savanna](https://github.com/Zymrael/savanna) and execute `make
  setup-env`. Make sure you clone the latest version of savanna which contains
  [this pull request](https://github.com/Zymrael/savanna/pull/9); this PR makes
  setup easier.
* Edit the Slurm script `finetune_launch.slurm` with your correct data and model
  configs, then submit the job with a simple sbatch.
* The finetuned savanna checkpoint needs to be converted to a vortex checkpoint
  to be usable for inference with the [dna-gen
  repository](https://github.com/evo-design/dna-gen). Use [this
  script](https://github.com/Zymrael/savanna/blob/main/tools/statedict_convert_checkpoint_to_vortex.py).
  After converting, the vortex checkpoint still needs one key rename. This repo
  contains a script called `hotfix_norm_scale.py` that does this.
