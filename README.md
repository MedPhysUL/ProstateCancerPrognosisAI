# ThePetaleProject


## Installation
In order to have all the requirements needed the user must do the following actions: 
- Install Pip(Nightly Build) version of dgl (https://www.dgl.ai/pages/start.html)
- Install torch 1.8.0 (https://pytorch.org/get-started/locally/)
- Install the remaining packages by running the following command:  
  `pip install -r settings/requirements.txt`
## Project Tree
```bash
├── create_experiments_recap.py      <- script that retrieves all models results from an experiment and creates
│                                       an html file to compare them
├── generale_learning_tables.py      <- script that creates a learning set and an holdout set
│                                       from a RAW learning table
├── generate_descriptive_analyses.py <- script that generates multiple statistics from a table 
│                                       in the PostgreSQL petale database.
├── generate_masks.py                <- script that generates train, valid and test masks from
│                                       a learning set (table) in the the PostgreSQL petale database.
├── run_warmup_experiments.py        <- script that runs models experiments on the WARMUP
│                                       learning set.
├── checkpoints                      <- torch models state dictionaries save by the EarlyStopper
├── csv                              <- csv files used for tables creation
├── experiments                      <- experiments script that can be called from root
├── hps                              <- json files used to store experiments hyperparameters
├── masks                            <- json files used to store train, valid and test masks 
│ 
├── records                          
│   ├── cleaning                     <- records generated by the DataCleaner module
│   ├── descriptive_analyses         <- records generated by generate_descriptive_analyses.py
│   ├── experiments                  <- records generated by the Evaluator module
│   ├── figures                      <- any important figures
│   ├── log_files                    <- any log files saved during experiments
│   └── tuning                       <- records generated by the Tuner module
│ 
├── sanity_checks                    <- scripts that validates modules behavior
│ 
├── settings
│   └── paths.py                     <- enum containing main project paths
│ 
├── src                              <- all project modules
│   ├── data
│   │   ├── extraction               <- modules related to data extraction from PostgreSQL
│   │   └── processing               <- modules related to data handling after extraction
│   ├── models                       <- learning models
│   │   ├── abstract models          <- modules related astract models architectures that cannot be used
│   │   │                               directly within the Petale framework
│   │   ├── blocks                   <- neural network architecture blocks
│   │   └── wrappers                 <- abstract classes used to wrap existing models and make them
│   │                                   work within the Petale framework
│   ├── recording                    <- recording module
│   ├── training                     <- modules related to models evaluation and tuning
│   └── utils                        <- modules related to visualization, score metrics, hyperparameters
│                                       and more
│ 
└── tables                           <- scripts used to generate new tables in PostgreSQL
    ├── intermediate            
    └── raw_learning
