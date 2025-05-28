# benchmark

Tools to calculate benchmarks, save and analyze results.

See `scripts` module for examples.

## Using MLFlow

This project uses MLFlow for tracking experiments, logging metrics, and saving parameters. The following instructions will guide you through how to set up and view your experiment results.

### 1. Setting Up the Environment
Make sure that you have MLflow installed in your Python environment. You can install it using pip:

```shell
pip install mlflow
```

### 2. Running the Experiment
Each benchmark script automatically logs metrics, the model version
and the current Git commit hash to MLflow.

#### 2.1 Default parameters 
You can run the script with default or custom directories for ground truth and predicted data.
For example:

```shell
python scripts/calculate_object_count_metrics.py
```

By default, the script looks for data in the following directories:
- Ground Truth: `benchmark/data_examples/object_counts/ground_truth`
- Predictions: `benchmark/data_examples/object_counts/predictions`

If you want to specify custom directories, use the command line arguments provided via `argparse`.

### 3. Viewing Results in MLFlow
If you omit the `--mlflow_dir` parameter, the default MLflow folder 
`mlruns` is created in the repository root.
To view the results of your experiment, launch the MLFlow UI with the following command:

```shell
mlflow ui
```

Once the server starts, open your browser and navigate to:

http://127.0.0.1:5000

Here, you can explore the logged metrics (such as mean IoU), parameters (like the first 5 characters of the Git commit hash), and other experiment-related data.

### 4. Experiment Output
- Git Commit Hash: The current Git commit hash is automatically logged as a parameter 
in MLFlow for version tracking.
- IoU Metric (for semantic segmentation case): 
The calculated Intersection over Union (IoU) is logged as the main evaluation metric.

### 5. Example Output
After running the script and opening the MLFlow UI, you will be able to see the logged metrics like this:

| Experiment | Git Commit | mean IoU |
|------------|-------------|----------|
| Exp1       | abcde       | 0.75     |
| Exp2       | fghij       | 0.80     |
