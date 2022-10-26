# DEGREE: A Data-Efficient Generation-Based Event Extraction Model

Code for our NAACL-2022 paper [DEGREE: A Data-Efficient Generation-Based Event Extraction Model](https://arxiv.org/abs/2108.12724).

## Environment
- Python==3.8
- PyTorch==1.8.0
- transformers==3.1.0 
- protobuf==3.17.3
- tensorboardx==2.4
- lxml==4.6.3
- beautifulsoup4==4.9.3
- bs4==0.0.1
- stanza==1.2
- sentencepiece==0.1.95
- ipdb==0.13.9


Note: 
- If you meet issues reated to rust when installing transformers through pip, this
[website](https://programmerah.com/solved-transformers-install-error-error-cant-find-rust-compiler-50679/) might be helpful

- Or you can reference the `env_reference.yml` for clearer installation

## Datasets

We support `ace05e`, `ace05ep`, and `ere`. 

### Preprocessing
Our preprocessing mainly adapts [OneIE's](https://blender.cs.illinois.edu/software/oneie/) released scripts with minor modifications. We deeply thank the contribution from the authors of the paper.

#### `ace05e`
1. Prepare data processed from [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event)
2. Put the processed data into the folder `processed_data/ace05e_dygieppformat`
3. Run `./scripts/process_ace05e.sh`

#### `ace05ep`
1. Download ACE data from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)
2. Run `./scripts/process_ace05ep.sh`

#### `ere`
1. Download ERE English data from LDC, specifically, "LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2", "LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2", "LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2"
2. Collect all these data under a directory with such setup:
```
ERE
├── LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
│     ├── data
│     ├── docs
│     └── ...
├── LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
│     ├── data
│     ├── docs
│     └── ...
└── LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
      ├── data
      ├── docs
      └── ...
```
3. Run `./scripts/process_ere.sh`

The above scripts will generate processed data (including the full training set and the low-resourece sets) in `./process_data`.

## Training

### DEGREE (End2end)

Run `./scripts/train_degree_e2e.sh` or use the following commands:

Generate data for DEGREE (End2end)
```Bash
python degree/generate_data_degree_e2e.py -c config/config_degree_e2e_ace05e.json
```

Train DEGREE (End2end)
```Bash
python degree/train_degree_e2e.py -c config/config_degree_e2e_ace05e.json
```

The model will be stored at `./output/degree_e2e_ace05e/[timestamp]/best_model.mdl` in default.

### DEGREE (ED)

Run `./scripts/train_degree_ed.sh` or use the following commands:

Generate data for DEGREE (ED)
```Bash
python degree/generate_data_degree_ed.py -c config/config_degree_ed_ace05e.json
```

Train DEGREE (ED)
```Bash
python degree/train_degree_ed.py -c config/config_degree_ed_ace05e.json
```

The model will be stored at `./output/degree_ed_ace05e/[timestamp]/best_model.mdl` in default.

### DEGREE (EAE)

Run `./scripts/train_degree_eae.sh` or use the following commands:

Generate data for DEGREE (EAE)
```Bash
python degree/generate_data_degree_eae.py -c config/config_degree_eae_ace05e.json
```

Train DEGREE (EAE)

```Bash
python degree/train_degree_eae.py -c config/config_degree_eae_ace05e.json
```

The model will be stored at `./output/degree_eae_ace05e/[timestamp]/best_model.mdl` in default.

## Evaluation

Evaluate DEGREE (End2end) on Event Extraction task 
```Bash
python degree/eval_end2endEE.py -c config/config_degree_e2e_ace05e.json -e [e2e_model]
```

Evaluate DEGREE (Pipe) on Event Extraction task 
```Bash
python degree/eval_pipelineEE.py -ced config/config_degree_ed_ace05e.json -ceae config/config_degree_eae_ace05e.json -ed [ed_model] -eae [eae_model]
```

Evaluate DEGREE (EAE) on Event Argument Extraction task (given gold triggers)

```Bash
python degree/eval_pipelineEE.py -ceae config/config_degree_eae_ace05e.json -eae [eae_model] -g
```

## Pre-Trained Models

| Dataset        | Model        | Model          | Model        |
| :------------- | :----------- | :------------- | :----------- |
| ace05e         | [DEGREE (EAE)](https://drive.google.com/file/d/1M6MMCGOE6sZeTXlmhwYts3iWhFuZk8TT/view?usp=sharing) | [DEGREE (ED)](https://drive.google.com/file/d/1Q0M_lf4jrQNiF6v-P1BxfMoBsU4GG7OW/view?usp=sharing) | [DEGREE (E2E)](https://drive.google.com/file/d/13iNZBVU2bGecQBkSIuNmR6Ob7fKEM1e9/view?usp=sharing) |
| ace05ep        | [DEGREE (EAE)](https://drive.google.com/file/d/1GbeHVvgX3x4FRMJjgvLExPWXWp-WM75C/view?usp=sharing) | [DEGREE (ED)](https://drive.google.com/file/d/1MZkeli2J12ThDatA-c_5CFXbf5IDr8nr/view?usp=sharing) | [DEGREE (E2E)](https://drive.google.com/file/d/1lWD8oOscw8l-HLiy2WYpM3QicKmND3KM/view?usp=sharing) |
| ere            | [DEGREE (EAE)](https://drive.google.com/file/d/1MFYIlFdIStSGl4mmYyRzhuLWBkkQyn4V/view?usp=sharing) | [DEGREE (ED)](https://drive.google.com/file/d/1HZCBYJpR2glMqCukxj4blHLkr8RRNKKE/view?usp=sharing) | [DEGREE (E2E)](https://drive.google.com/file/d/16Mv-C2ZpQS6K3IqbgYMFxAibtOLD9x-t/view?usp=sharing) |


## Citation

If you find that the code is useful in your research, please consider citing our paper.

    @inproceedings{naacl2022degree,
        author    = {I-Hung Hsu and Kuan-Hao Huang and Elizabeth Boschee and Scott Miller and Prem Natarajan and Kai-Wei Chang and Nanyun Peng},
        title     = {DEGREE: A Data-Efficient Generative Event Extraction Model},
        booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
        year      = {2022},
    }

## Contact

If you have any issue, please contact I-Hung Hsu at (ihunghsu@usc.edu) or
Kuan-Hao Huang at (khhuang@cs.ucla.edu).
