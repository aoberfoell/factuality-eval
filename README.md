# FactualityPrompt
## 1. Setup 
1. in console: `pip install -r requirements.txt`
    - download spacy model with `python -m spacy download en_core_web_sm`
2. Download Wikipedia processed dump (knowledgesource.json) from [KILT-github](https://github.com/facebookresearch/KILT#kilt-knowledge-source) into `data` directory (Refer to their repository for citation details)
```bash
  mkdir data
  cd data
  wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
3. Create the DB file *kilt_db.db* from Wikipedia dump by running from dir `fever_athene/src`:

```bash
  python scripts/build_db_kilt.py ../../data/knowledgesource.json ../../data/kilt_db.db
```

4. Set paths in `src/const.py`

## 2. Create generations
The file containing the generations is a *.jsonl* file with the following keys:
- *id*: id of the example
- *prompt*: the prompt for which the text was generated
- *text*: the generated text including the prompt?
1. Set params in *run_generations.py*
2. `python run_generations.py`
    - file generations is saved to *GEN_DIR*

## 3. Run evaluation script
Running any of the scripts below will save corresponding metric results into a file named `$GEN_TO_EVALUATE_NAME_results.jsonl` (`$GEN_TO_EVALUATE_NAME` refers to the file containing generations that you are trying to evaluate).

### Factuality Metric (Hallucinated NE Error, Entailment Ratio)

```bash
PROMPT_TYPE=factual
GEN_FILENAME=gens.jsonl
python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_FILENAME}
```

### Repetition

```bash
GEN_FILENAME=gens.jsonl
python src/repetition.py ${GEN_FILENAME} --final
``` 

### Diversity

1. First obtain multiple generation files from your LM with different seed. In our paper, we used 10 random seeds, but you can use your own choice of seed count. **If you are evaluating greedy, there is NO NEED to generate multiple seed, because all seed will result in same generation. Simply use 1 generation file.**

2. Then run the below script:
```bash
GEN_DIR=directory-containing-multi-seed-generation-files
FILE_TEMPLATE=shared-string-between-multiple-seed-generation
python src/distinct_n.py --gen_dir ${GEN_DIR} --file_template ${FILE_TEMPLATE} --number_of_seeds 10
```

Illustration of `FILE_TEMPLATE`:
* Let's assume your generation files are named as follows: factual_gen_seed1.jsonl, nonfactual_gen_seed1.jsonl, factual_gen_seed2.jsonl, nonfactual_gen_seed2.jsonl,...
* Then, your `FILE_TEMPLATE` will be "gen_seed"