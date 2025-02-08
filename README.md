# FIGA: Enhancing Individual Fairness Testing Through Feature Importance-Guided Genetic Algorithm
**Authors**: _Hussaini Mamman, Shuib Basri, Abdullateef Oluwagbemiga Balogun, Abdul Rehman Gilal, Shamsudden Adamu, Aliyu Garba, and Luiz Fernando Capretz._

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hmamman/FIGA.git
   cd FIGA
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Running Fairness Testing

### Command-Line Arguments

The script accepts the following arguments:

- `--dataset_name`: (string) Name of the dataset to use in the experiment. The default is `'census'`.
  - Example: `--dataset_name census`

- `--sensitive_name`: (string) Name of the sensitive attribute for fairness testing (e.g., `sex`, `age`, `race`). The default is `'age'`.
  - Example: `--sensitive_name sex`

- `--classifier_name`: (string) Name of the classifier to use (e.g., `mlp`, `dt`, `rf`, ect.). The default is `'dt'`.
  - Example: `--classifier_name svm`

- `--max_allowed_time`: (integer) Maximum time in seconds for the experiment to run. The default is `3600` seconds (1 hour).
  - Example: `--max_allowed_time 3600`

### Example Usage

To run the FIGA framework:
```bash
python python ./figa_tutorial/figa.py --classifier_name dt --dataset_name bank --sensitive_name age --max_allowed_time 3600
```

To run a specific benchmarking approach included in this repository:
```bash
python ./baseline/expga/expga.py --classifier_name dt --dataset_name bank --sensitive_name age --max_allowed_time 3600
```

You can also run all experiments for an approach (figa, expga, sg, and fairbs) using main.py file:
```bash
python ./main.py --approach_name fairbs --max_allowed_time 3600  --max_iteration 5  
```

