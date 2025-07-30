# Savanna finetuning guide

This repo contains instructions (in this README) and scripts, configs, etc, for
finetuning Evo 2 7B using the [savanna](https://github.com/Zymrael/savanna)
repository. I haven't tried finetuning Evo 2 1B or 40B using savanna, but they
*should* work similarly in principle; you can build off of these scripts.

* Process your dataset from JSONL (*must* be JSONL; TXT doesn't work reliably)
  to BIN and IDX files using [this
  script](https://github.com/Zymrael/savanna/blob/main/tools/preprocess_data.py).
  For the tokenizer type, use CharLevelTokenizer. You can run the script from
  the terminal using a command like the one below. See the [section on
  packing](#packing) below for more on the `--enforce-sample-length` flag.

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
  repository](https://github.com/evo-design/dna-gen).
  * Use [this
    script](https://github.com/Zymrael/savanna/blob/main/tools/statedict_convert_checkpoint_to_vortex.py).
    To use it, first `mv` it out of `tools/` and into the root directory. Then,
    scroll down to `main()` and edit `checkpoint_path` with the savanna PT file
    to be converted (would look something like
    `/storage/evo2-human-finetune/global_step100000/mp_rank_00_model_states.pt`),
    `config_path` with the model config YAML you used (a copy will be saved in
    `/storage/evo2-human-finetune/global_step100000/configs/the-config-name-i-used.yml`),
    `new_checkpoint_path` with the directory where you want the vortex
    checkpoint to be saved (something like
    `/storage/evo2-human-vortex-finetune/epoch10/`), and `iteration` with
    whatever number you like (it's just used for naming the file; let's say you
    go with `10`). The vortex checkpoint will be saved to
    `/storage/evo2-human-vortex-finetune/epoch10/iter_10.pt`. The script
    typically executes in under 10 seconds.
  * After converting, the vortex checkpoint still needs one key rename. This
    repo contains a script called `hotfix_norm_scale.py` that does this.

## Packing

* As a recap, the command for preprocessing your data from JSONL to BIN and IDX
  files is:

```bash
python tools/preprocess_data.py --input data/raw/data_for_train.jsonl --output-prefix data/run/padded4096_data/train --workers 8 --enforce-sample-length 4096 --tokenizer-type CharLevelTokenizer
```

* Notice that without the `--enforce-sample-length` flag, you get [contiguous
  packing](https://huggingface.co/blog/sirluk/llm-sequence-packing#the-solution-sequence-packing),
  whereas with the flag, you get right padding. You might want one or the other
  depending on your use case.
* For a concrete example, if you want to *condition* your prompts to Evo 2, you
  will have sequences like `[CLADE_TYPE_A]ACGTG...`, `[CLADE_TYPE_B]CTCTA...`,
  `[CLADE_TYPE_C]GCTCA...`, and so on. The idea is that sequences from Clade A
  look meaningully different from sequences in Clade B. If you prepend a clade
  token, Evo 2 will learn to associate the clade token with the distribution of
  sequences that follow it. After finetuning, if you prompt Evo 2 with
  `[CLADE_TYPE_A]`, it will generate a sequence similar to the sequences it has
  seen following Clade A.
* Sidenote: for concreteness, the tokens `[CLADE_TYPE_A]`, `[CLADE_TYPE_B]`, and
  `[CLADE_TYPE_C]` can be `1`, `2`, and `3`, or any of the 256 "[extended
  ASCII](https://en.wikipedia.org/wiki/Extended_ASCII)" characters. As [this
  `savanna`
  code](https://github.com/Zymrael/savanna/blob/80377fe74b7acd41253e03cba3750a5fcd57e32b/savanna/tokenizer/tokenizer.py#L277)
  shows, Evo 2 uses a custom `CharLevelTokenizer` that simply maps characters to
  their `np.uint8` (i.e., 2 ** 8 = 256) representations, and the [Evo 2
  config](https://github.com/ArcInstitute/evo2/blob/main/evo2/configs/evo2-7b-1m.yml)
  gives Evo 2 a comfortable vocabulary size of 512 (actually twice the room
  `CharLevelTokenizer` needs).
* To make sure that the clade type token is always associated with the
  nucleotide sequences, you want right padding. Otherwise, the clade indicator
  token might get left behind in a previous sequence.
* If you don't want to prompt with a start token, and just want to finetune Evo
  2 in the sense of continued pretraining on new data (for example, if you're
  finetuning the model on a new batch of metagenomic data), use contiguous
  packing for training efficiency. Right padding fills unused positions with
  `[PAD]` tokens on which loss is masked, wasting compute.
* Here's a schematic that helps visualize what happens with contiguous and right
  padding.

![](/assets/packing_schematic.png)
