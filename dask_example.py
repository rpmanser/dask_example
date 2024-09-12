import argparse
from pathlib import Path

import dask
import dask_jobqueue
import numpy as np
import scipy
import xarray as xr
from herbie.fast import FastHerbie


def main():
    parser = argparse.ArgumentParser(
        "Download GEFS precipitation forecasts, then calculate the "
        "daily probability of precipitation across CONUS."
    )
    parser.add_argument(
        "init", type=str, help="Forecast initialization date formatted as YYYY-MM-DD"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of workers to use in the Dask cluster.",
    )
    args = parser.parse_args()
    init = args.init
    n_jobs = args.n_jobs

    save_dir = Path("/lustre/scratch/rmanser")

    # Download GEFS member precipitation forecasts
    members = xr.DataArray(np.arange(1, 11, dtype=int), name="member", dims="member")

    files = list()
    for m in members.values:
        model = FastHerbie(
            [init],
            fxx=range(6, 846, 6),
            model="gefs",
            product="atmos.5",
            save_dir=save_dir,
            member=int(m),
            n_jobs=n_jobs,
        )
        files.append(model.download("APCP"))

    # Start a dask cluster for opening and processing GEFS forecasts
    cluster = dask_jobqueue.SLURMCluster(
        cores=1,
        memory="3GB",
        queue="nocona",
        job_extra_directives=["--nodes 1"],
    )
    client = dask.distributed.Client(cluster)
    cluster.scale(n_jobs)

    # Read GEFS forecasts, align along lead time and member dimensions
    datasets = list()
    for mfiles in files:
        ds = xr.open_mfdataset(
            mfiles,
            concat_dim="step",
            combine="nested",
            engine="cfgrib",
            parallel=True,
        ).rename({"number": "member"})
        datasets.append(ds)

    ensemble = xr.concat(datasets, dim=members)

    # Calculate daily precipitation probabilities
    daily_precip = ensemble["tp"].resample(step="24h").sum()
    daily_probs = daily_precip.where(daily_precip <= 0.254, 1.0, 0.0).mean(dim="member")
    daily_probs = xr.apply_ufunc(
        scipy.ndimage.uniform_filter,
        daily_probs,
        kwargs=dict(size=24, mode="mirror"),
        dask="parallelized",
    ).compute()

    daily_probs.to_netcdf(save_dir / f"gefs_24h_precip_prob_{init}.nc")

    for mfiles in files:
        for f in mfiles:
            f.unlink()

    gefs_dir = files[0][0].parent
    idx_files = gefs_dir.glob("*.idx")
    for idx in idx_files:
        idx.unlink()

    cluster.scale(0)
    client.close()


if __name__ == "__main__":
    main()
