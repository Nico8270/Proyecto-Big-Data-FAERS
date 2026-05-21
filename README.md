# FAERS Pharmacovigilance Analytics Platform

**FAERS Pharmacovigilance Analytics Platform** - A scalable processing pipeline using Hadoop and MapReduce

## Project Structure

```
faers-pharmacovigilance/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies (EDA)
├── requirements_hadoop.txt     # Hadoop dependencies (mrjob)
├── .gitignore
├── .env.example                # Environment variables template
│
├── docs/                       # Documentation
│   ├── 1_introduction.md      # Full project introduction
│   ├── 2_project_profile.md   # Project profile from Word doc
│   ├── README.md              # Project overview
│   ├── USAGE.md               # Quick usage guide
│   ├── PROJECT_SUMMARY.md     # Executive summary
│   ├── OUTPUTS_EXAMPLES.md    # Output examples
│   ├── informe.md             # FAERS dataset documentation
│   ├── hadoop_README.md       # Hadoop guide
│   └── SUMMARY.md             # Technical summary
│
├── src/                        # Source code
│   ├── main.py                # Main entry point
│   ├── eda/                   # Exploratory Data Analysis
│   │   ├── 00_eda_simple.py   # Simplified EDA script
│   │   └── run_all.py         # EDA runner
│   └── hadoop/mapper/         # Hadoop/MapReduce jobs
│       ├── top_drugs.py       # Top drugs mapper
│       └── run_faers_pipeline.py  # Pipeline runner
│
├── config/                    # Hadoop configuration
│   ├── core-site.xml
│   ├── hdfs-site.xml
│   ├── mapred-site.xml
│   └── yarn-site.xml
│
├── docker/                    # Docker deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-entrypoint.sh
│
├── data/                      # FAERS dataset (Q4 2025)
│   ├── DEMO25Q4.txt
│   ├── DRUG25Q4.txt
│   ├── REAC25Q4.txt
│   └── ... (7 txt files)
│
├── outputs/                   # Generated outputs
│   ├── eda_results/           # EDA graphs and CSVs
│   └── mapreduce_results/     # MapReduce outputs
│
├── tests/                     # Test suite
└── run.bat                    # Windows runner with menu
```

---

## Quick Start

### Using run.bat (Recommended)

```bash
run.bat
```

Menu options:
1. Install Technologies
2. Run EDA Pipeline
3. Run Hadoop MapReduce
4. Run All Pipelines
5. Setup Virtual Environment
6. Exit

### Command Line

```bash
# EDA
python src/main.py --eda

# Hadoop
python src/main.py --hadoop

# All
python src/main.py --all
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/1_introduction.md` | Full project introduction |
| `docs/2_project_profile.md` | Project profile |
| `docs/README.md` | Project overview |
| `docs/USAGE.md` | Usage guide |
| `docs/PROJECT_SUMMARY.md` | Executive summary |
| `docs/hadoop_README.md` | Hadoop guide |