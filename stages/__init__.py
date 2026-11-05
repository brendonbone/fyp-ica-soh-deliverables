"""
stages: one module per calculation, run in order by run_results.py.

Each module

  * starts with a short docstring: the question it answers, the inputs it
    reads, the files it writes, how long it takes;
  * keeps all paths under src.config.RESULTS_DIR (results/) and the raw data
    under src.config.PKL_DIR;
  * has a main() guarded by `if __name__ == "__main__"`, so it also runs on
    its own:  python -m stages.s04_feature_pool [options];
  * writes result files only (JSON / CSV / parquet). No figures: the report
    items in report_items/ draw from the result files.

Long searches write a progress file after every unit and skip finished units
on a rerun; the ones marked shardable take --shard I --n-shards N for a
cluster job array and --merge to combine the shards (see cluster/README.md).
"""
