# Running the slow stages on Katana (UNSW)

The searches of stages 14-17 take hours to days on one machine. On Katana each
runs as a PBS job array, one shard per subjob; the shards are merged and
analysed back on the laptop. Full docs: https://docs.restech.unsw.edu.au

## Hosts

| Alias | Host | Use for |
|---|---|---|
| `katana` | `katana.restech.unsw.edu.au` | submitting and monitoring jobs, editing code |
| `kdm` | `kdm.restech.unsw.edu.au` | **all** file transfers, tarring, unzipping |

Never run compute on the login node (4 GB per user, killed without warning)
and never push transfers through `katana`: that is what `kdm` is for. Home
(`/home/$USER`, 10 GB, backed up) holds the repository; scratch
(`/srv/scratch/$USER`, 128 GB, not backed up) holds the data and the venv.
If home is at its quota, put the checkout on scratch too (the 2026-10-11 runs
did) and submit from there.

## One-time setup

```bash
ssh katana
git clone <repo url> ~/fyp-ica-soh-deliverables   # or git archive HEAD on the laptop, unpacked via kdm
mkdir -p /srv/scratch/$USER/fyp-data               # the Severson pickles go here (see ../README.md, "Data")
ln -s /srv/scratch/$USER/fyp-data ~/fyp-ica-soh-deliverables/data
module load python/3.13.2                          # newest on Katana; the laptop runs 3.14
python3 -m venv /srv/scratch/$USER/envs/fyp313
source /srv/scratch/$USER/envs/fyp313/bin/activate
pip install -r ~/fyp-ica-soh-deliverables/cluster/requirements-katana.txt
```

`data/` is gitignored, so the symlink is safe; `src.config` resolves the
pickles as `data/Severson Dataset/Severson pkl/batch{1,2,3}.pkl`, and setting
`ICA_SOH_DATA=/srv/scratch/$USER/fyp-data` in the job does the same without a
link. Only stages 1, 2, 3 and 10 read the pickles; the sharded stages run from
`results/features_expanded.parquet`, which is committed.

`requirements-katana.txt` pins the laptop's versions of the packages `src/` and
`stages/` import (no matplotlib: the report is drawn on the laptop). The PBS
scripts activate `/srv/scratch/$USER/envs/fyp313`; pass `-v VENV=<path>` to
use another venv.

## A single stage

```bash
cd ~/fyp-ica-soh-deliverables && git pull
qsub -v STAGE=s02_extract_features -N s02 cluster/job_template.pbs
qstat -u $USER                                     # watch it
tail -f cluster/logs/s02.live
```

`ARGS` passes extra flags to the stage: `-v STAGE=s08_history_features,ARGS="--jobs 2"`.

## A sharded search (stages 14, 15, 16) or the stage-17 tasks

1. Run stage 8 on the laptop first: it writes `results/features_history.parquet`
   (gitignored), which the array jobs read instead of recomputing it (see
   "FYP_HISTORY_TABLE" below). Copy it over through `kdm`:

   ```powershell
   scp results/features_history.parquet kdm:fyp-ica-soh-deliverables/results/
   ```

2. Submit from the repository root (`-N` names the log files):

   ```bash
   qsub -J 0-7  -v STAGE=s14_history_components,NSHARDS=8         -N s14 cluster/eng2x_array.pbs
   qsub -J 0-7  -v STAGE=s15_level_subsets_history_rows,NSHARDS=8 -N s15 cluster/eng2x_array.pbs
   qsub -J 0-31 -l walltime=03:00:00 -v STAGE=s16_history_subsets,NSHARDS=32 -N s16 cluster/eng2x_array.pbs
   qsub -J 0-10 -v STAGE=s17_partition_elimination,NSHARDS=0      -N s17 cluster/eng2x_array.pbs
   ```

   Subjob `I` runs `python -m stages.<STAGE> --shard I --n-shards N --n-jobs $NCPUS`
   and writes its own `results/<stem>_progress_shardIIofNN.json` after every
   subset. `NSHARDS=0` passes `--task I` instead: stage 17's eleven
   mode x partition tasks, each with `results/eng25_progress_<mode>_<partition>.json`.
   Live output goes to `cluster/logs/<name>_<I>.live`. A killed array is
   resubmitted as it was: every subjob skips the units already in its
   progress file. An 8-core subjob runs these fits 2.5-3.2x slower than the
   24-thread laptop, so the gain comes from running many at once.

## Merging and analysing on the laptop

```powershell
scp "kdm:fyp-ica-soh-deliverables/results/*_progress_*.json" results/
python -m stages.s16_history_subsets --merge      # shard files -> eng23_grouped_history_subsets.{csv,json}
python -m stages.s16_history_subsets --analyse    # or python run_results.py: the stage sees its table and skips the search
python -m stages.s17_partition_elimination --report
```

`--merge` unions the progress files; a unit scored twice must agree within
`MERGE_TOL = 1e-6` (`src/subset_search.py`), and any difference above 1e-12
is logged in the merged JSON (`merge_conflicts`, with both hosts). `--analyse`
also needs the gitignored `results/features_soc_windows.parquet` (stage 10)
for the SoC windows, so it is run on the laptop.

## FYP_HISTORY_TABLE

The history columns are an OLS matrix product (`src/features_temporal.py`),
and the BLAS summation order depends on the CPU's vector width: on Katana's
AVX-512 nodes they differ from the laptop in the last bits (relative ~1e-12),
which moved a 63-column forest's dev-CV RMSE by 2e-7. So the array jobs never
recompute the table: `eng2x_array.pbs` exports
`FYP_HISTORY_TABLE=results/features_history.parquet` and refuses to start
without the file; `src.subset_search.history_table` then reads it (the
laptop's table bit for bit, written by stage 8 via `write_history_table`,
which checks the parquet round trip) instead of calling `add_temporal_features`.
Unset, as on the laptop, the table is computed as in stage 8. The host that
scored each subset is kept in the merged tables (`host`).

## Useful

```bash
qsub -I -l select=1:ncpus=4:mem=16gb -l walltime=02:00:00   # interactive node
qstat -f <jobid>                                            # why is it queued
qdel <jobid>
qstat -x -f <jobid> | grep -i resources_used                # what it actually used
```

Resources: ~4 GB of memory per core; much more, or walltime over 12 h (the cap
for general users), means a longer wait. Transfers: always through `kdm`, tar
first (one big file moves in seconds, loose files at ~20 files/s), inside
`tmux` on `kdm` for anything long. From PowerShell:

```powershell
tar -czf data.tar.gz -C "C:/path/to" "Severson Dataset"
scp data.tar.gz kdm:/srv/scratch/z5482679/fyp-data/
ssh kdm "cd /srv/scratch/z5482679/fyp-data && tar -xzf data.tar.gz && rm data.tar.gz"
```
