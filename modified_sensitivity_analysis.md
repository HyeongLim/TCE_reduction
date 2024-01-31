---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python [conda env:UQsensitivity]
    language: python
    name: conda-env-UQsensitivity-py
---

```python
import os
import glob
import re
import numpy as np
import arviz as az
import xarray as xr
import pandas as pd
import datatree as dt
import seaborn as sns
import matplotlib.pyplot as plt
import dask.delayed
from dask.distributed import Client
import SALib as sa
import nesttool
import tce_models.output_processing as output_processing
import tce_models.prior_sampling as prior_sampling
```

```python
from tce_models import ureg
import pint
import pint_xarray
pint_xarray.unit_registry = ureg
from tce_models.quantities_of_interest import compute_fluxes_raw
```

<!-- #region -->
# Model inputs and outputs

The Crunchflow models uses a large number geochemical parameters.
Some of the parameters are kinetic parameters, that is, they determine how fast reactions happen (rate constants of mineral dissolution/precipitation reactions, half-saturation constants).
Other parameters are thermodynamic parameters that determine the distribution of chemical species at equilibrium (e.g., equilibrium constants of aqueous complex formation, surface complexation, and mineral dissolution/precipitation).
Another category of parameters determine the initial and boundary conditions (concentrations and mineral fractions).

We treat most of the thermodynamic parameters as constants and do not account for their uncertainty in the sensitivity analysis (values are provided in the Crunchflow input file and geochemical database).
The equilibrium constants of FeS and goethite dissolution/precipitation are an exemption.
In contrast, most of the kinetic parameters will be accounted for in the sensitivity analysis (see table below).

In addition to the parameters listed in the table below, parameters related to TCE reduction reaction (specific surface area $A_m$ of the reactive minerals, kinetic rate constant $k_m$ and initial mineral fractions of the reactive minerals) are considered in the sensitivity analysis.


# Define the prior distributions

Prior distributions of parameters are defined using the distribution functions in the `stats` module of SciPy.
The type of distribution and the distributions' parameters are given in the table below (read from the file [`priors.csv`](../crunchflow_models/full_geochemistry/priors.csv)).
<!-- #endregion -->

```python
priors = pd.read_csv(
        "../crunchflow_models/full_geochemistry/priors_modified.csv",
        header=0,
        skipinitialspace=True,
        dtype={"references": str},
        comment="#"
)
```

```python
## Format the table for printing in a Quarto document
#table_formatters = {
#    "parameter": {
#        "half_saturation_constant": "$K_s$",
#        "log10_dissolution_rate": r"$\log_{10}(k_m)$",
#        "log10_equilibrium_constant": r"$\log_{10}(K_{eq})$",
#    },
#}
#
#def format_reference_for_quarto(references, sep=";"):
#    ref_list = references.split(sep)
#    return ";".join([f"@{ref}" if ref!= "" else ref for ref in ref_list])
#
#priors.replace(table_formatters)
#priors.references.fillna("").apply(format_reference_for_quarto)
```

```python
# Print the table
priors.set_index(["parameter", "index"])
```

## Parameter transformations
Some input parameters need to be transformed before they can be input into the model. For example, prior distributions of some parameters are defined on a log scale, but the model requires parameters to be define on a non-logscale, so the exponential has to be applied first. Another transformation accounts for conversion between the natural logarithm and $\log_{10}$.
These transformations are applied in the function `transform_params` of the [`prior_sampling`](../tce_models/prior_sampling.py) module.

## Samples of dependent parameters produced by another model as inputs
The parameter samples related to TCE reduction are generated differently than all other parameters.
These samples are read from a file, and they originate from the posterior distribution of [another model](https://gitlab.com/astoeriko/hierarky), where I estimate TCE reduction rate constants and specific surface areas based on laboratory data.
The initial idea was that independently drawn samples of specific surface areas and TCE reduction rate constants would likely lead to effective rate constants that are too large or too small.
Specific surface areas and rate constants appear as a product in the reaction rate, so observed rates can be explained by many combinations of rate constants and surface areas.
However, if both the specific surface area and the rate constant are large the overall rate might be much larger than the observed one.
In the same way, while the reaction rate constant of an individual mineral may be large, it is unlikely that *all* minerals can have large rate constants.
I hypothesized that by coniditoning with observed effective rate constants I would obtain samples of the specific surface areas and reduction rate constants that potentially show a dependence between the different parameters, but that are consistent with the data.

Because of this approach, only sensitivity analysis methods that do not assume independence of the input parameters should be applied.
Also, only methods that can be used with random samples (instead of sampls generated with a specific design) can be used.

However, the actual correlations between the parameters in the posterior distribution are not large.
If we need to run more simulations, we might opt for generating independent samples of all input parameters (instead of using the correlated samples generated by the other model). This would allow for using a wider number of sensitivity analysis methods.

# Set up the model runs

The wrapper functions for setting up the Crunchflow model are mostly contained in the module [`prior_sampling`](../tce_models/prior_sampling.py).
For submitting the model to the client, I need a funciton that takse only the sample/realization number as an input, so for convenience I define it below, using variables defined in the notebook as global variables inside the function. (It would probably be a cleaner solution to create a function that gets passed all variables as arguments, and then create a function with just one argument with `functools.partial`.)

```python
home = os.environ["HOME"]
base_path = os.path.join(home, "git-repos/tce_models/outputs/2024-01-29_full_geochemistry")
template_base_path = os.path.join(home, "git-repos/tce_models/crunchflow_models/full_geochemistry/")
crunch_path = os.path.join(home, "git-repos/crunchtope-dev/source/CrunchTope")
```

```python
n_realizations = 5 # 8000
model_name = "full_geochemistry"
template_paths = prior_sampling.make_template_paths(template_base_path, model_name)
idata = az.from_netcdf("../outputs/trace_explicit_minerals.nc").stack(sample=("chain", "draw"))
```

```python
idata
```

```python
def run_full_geochemistry_model(realization):
    model_run_id = f"realization_{realization}"
    # If I define the distributions globally, or if pass them as an argument to the function instead
    # of defining them within the function, I end of with the same numbers in different realizations.
    # This does not happen if I run the function several times within the notebook, but it does happen
    # if I run it several times with `client.map`.
    dists = prior_sampling.define_distributions(priors)
    params = prior_sampling.generate_realization(dists)
    #params = prior_sampling.add_tce_param_samples(params, idata.posterior.isel(sample=realization))
    transformed_params = prior_sampling.transform_params(params)
    model_dir = prior_sampling.run_model(
        transformed_params,
        base_path=base_path,
        model_run_id=model_run_id,
        crunch_path=crunch_path,
        template_paths=template_paths,
        timeout=60 * 90,
        model_name=model_name,
    )
    params_ds = prior_sampling.params_dict_to_xarray(transformed_params, model_run_id)
    params_ds.to_netcdf(os.path.join(model_dir, "parameters.nc"))
    return params_ds
```

# Start a dask client and submit the model runs to the client

```python
client = Client(n_workers=20, threads_per_worker=1, memory_limit="50GB")  # set up local cluster on your laptop

client
```

# Start the Crunchflow model runs

Uncomment the code in the following cells in order to run the model.

```python
futures = client.map(
    run_full_geochemistry_model,
    range(n_realizations),
    pure=False,
)
```

```python
 result = client.submit(
     xr.concat, futures, pd.RangeIndex(n_realizations, name="realization"), coords="all"
 )
```

```python
 params_ds = client.gather(result)
```

```python
 params_ds.to_netcdf(os.path.join(base_path, "all_params.nc"))
```

# Read in data and combine it into one dataset

The model outputs are quite large, and I was always running into problems if I try to read in all realizations at once.
Thus, I decided to never read in all the data at once. Instead, I compute the quantities of interest for each dataset individually.
As the quantities of interest are a much smaller array, I can then easily concatenate them afterwards.
I read in the datasets lazily with dask, which also enables me to easily compute the quantities of interest in paralell on a dask client.

Reading in the data is the slowest step, so overall this still takes some time (I should probably save the quantities of interest, so I can quickly read them in again afterwards).

```python
# Read in parameters from files
param_paths = glob.glob(os.path.join(base_path, "realization_*/parameters.nc"))
output_paths = [path.rstrip("/parameters.nc") for path in param_paths]
model_run_ids = [re.match(".*/(realization_[0-9]*)", path).groups()[0] for path in output_paths]
```

```python
params_ds = xr.open_mfdataset(
    paths=[os.path.join(path, "parameters.nc") for path in output_paths],
    combine='nested',
    concat_dim="realization"
)["__xarray_dataarray_variable__"].rename("parameters")
```

```python
params_ds
```

```python
def read_realization_i(model_run_id, base_path):
    model_dir = os.path.join(base_path, model_run_id)
    tree = prior_sampling.read_output_data(model_dir, model_run_id, x_units="m", time_units="yr",)
    return tree


def flatten_outputs(tree):
    stacked_data = output_processing.stack_output_data(tree)
    return output_processing.flatten_datatree(stacked_data)


def save_model_outputs_as_zarr(model_run_id, base_path, model_name):
    "Read in Crunchflow output files and save the resulting datatree as zarr file"
    tree = read_realization_i(model_run_id, base_path)
    model_dir = os.path.join(base_path, model_run_id)
    tree.to_zarr(os.path.join(model_dir, f"{model_name}.zarr"))


def save_model_outputs_as_netcdf(i, model_name):
    "Read in Crunchflow output files and save the flattened dataset as netcdf file"
    tree = read_realization_i(i)
    flattened = flatten_outputs(tree)
    flattened.to_netcdf(os.path.join(model_dir, f"{model_name}.nc"))
```

```python
@dask.delayed
def read_model_outputs_from_zarr(path, chunks=None):
    tree = dt.open_datatree(path, engine="zarr", chunks=chunks)
    # TODO: Do I need to flatten the datatree? It takes quite a bit of time.
    # The reason I flattened the datatree was that I cannot concatenate the datatrees
    # directly. However, I now do the concatenation only at the very end anyway,
    # with a dataset that contains the quantities of interest.
    ds = flatten_outputs(tree)
    ds["/model_run_id"] = ds["/model_run_id"].astype("str")
    return ds # add .persist()
```

```python
# Read in text files for each model run and save as zarr separately.
# Note: This needs to be done only once after the Crunchflow model runs, but it takes a
# long time because reading in all the text files is slow.

# futures = client.map(
#     save_model_outputs_as_zarr,
#     model_run_ids,
#     base_path=base_path,
#     model_name="full_geochemistry"
#)
```

```python
@dask.delayed
def fill_dataset(ds, reference_ds):
    empty = xr.full_like(reference_ds, fill_value=np.nan)
    return empty.combine_first(ds).assign({"/model_run_id": ds["/model_run_id"]})


@dask.delayed
def create_concat_coord(ds):
    return (
        ds.rename_vars({"/model_run_id": "model_run_id"})
        .set_coords("model_run_id")
        .expand_dims("model_run_id")
        .rename_dims(model_run_id="realization")
    )


def append_to_zarr(ds, path):
    return ds.to_zarr(path, append_dim="realization")
```

```python
# Read the model outputs from zarr again
paths = [os.path.join(basepath, f"{model_name}.zarr") for basepath in output_paths]
datasets = [read_model_outputs_from_zarr(p) for p in paths]

# Find out which model run has the most complete coordinates (that is, find one that finished)
# Read in the corresponding dataset as a reference for the coordinates
reference = datasets[0].compute()
#reference = create_concat_coord(reference).compute()

# Write reference dataset to zarr
#path_combined = os.path.join(base_path, "all_outputs.zarr")
#create_concat_coord(reference).compute().to_zarr(path_combined)
```

```python
# For each other dataset:
# Fill with NaN if necessary
filled = [fill_dataset(ds, reference) for ds in datasets]
# Set dimension to append along
with_concat_dim = [create_concat_coord(ds) for ds in filled]
# Write dataset to zarr in "append_dim" mode
# Appending does not work because the slashes in the var names mess it up
# (The slashed lead to several groups being created, and then variables I want
# to append to (having the full path as theri name) cannot be found.)
# stores = [append_to_zarr(ds, path_combined) for ds in with_concat_dim[1:]]
```

```python
# Combine all datasets into a single one
# This leads to the whole dataset being loaded into memory
#result = dask.delayed(xr.concat)(datasets, pd.Index(model_run_ids, name="model_run_id"), coords="all")
```

# Define quantities of interest for the sensitivity analysis

For now, I test the following quantities of interest in order to reduce multidimensional model outputs to scalar quantities:
- Diffusion flux of TCE (and the conservative tracer) out of the domain at the boundary with the aquifer,
- the reaction rates of microbial iron and sulfate reduction, integrated over time and the spatial domain of the model,
- the change in reactive mineral content at the end of the simulation compared to the initial value (averaged over the domain)
- the contribution of the different reactive minerals to TCE removal (measured by the fraction of the spatially and temporally integrated reaction rate of a single mineral compared to the total rate).

```python
from tce_models.quantities_of_interest import compute_qoi, compute_cumulative_fluxes, compute_relative_cumulative_tce_flux
```

```python
qois_delayed = [compute_qoi(delayed) for delayed in with_concat_dim]
concatenated = dask.delayed(xr.concat)(qois_delayed, "realization")
```

```python
qoi = concatenated.compute()
```

```python
fluxes = qoi["diffusion_fluxes"]
fluxes
cumulative_fluxes = compute_cumulative_fluxes(fluxes)
relative_cumulative_tce_flux = compute_relative_cumulative_tce_flux(fluxes)
```

```python
np.log(relative_cumulative_tce_flux).plot.hist();
```

```python
qoi["integrated_dirb_rate"].pint.to("mole/m^2").plot.hist()
```

```python
qoi["tce_removal_contributions"].isel(mineral_reaction=2).plot.hist()
```

```python
idx_realization = qoi["average_mineral_content_change"].isel(mineral=0).argmax()
```

```python
data = (
    qoi["average_mineral_content_change"]
    .where(lambda x: x < 1e20)
    .to_pandas()
    .unstack()
    .rename("change in mineral content")
    .reset_index()
)
```

```python
sns.displot(
    data.where(data["mineral"] == "Pyrite").dropna(),
    x="change in mineral content",
    col="mineral",
    kind="kde",
    log_scale=True
)
```

```python
sns.displot(
    data.where(data["mineral"] != "Pyrite").dropna(),
    x="change in mineral content",
    col="mineral",
    kind="kde",
    log_scale=False
)
```

# How long did the simulations take? How many finished?

```python
sns.displot(qoi["simulated_time"].to_series(), rug=True, stat="percent")
```

```python
finished = (qoi["simulated_time"] > 59).rename("finished")
```

```python
finished.mean("realization")
```

```python
def read_run_time(path):
    path = os.path.join(path, "stdout")
    try:
        with open(path, "r", errors='surrogateescape') as f:
            line_with_runtime = f.readlines()[-2]
        return parse_runtime(line_with_runtime)
    except FileNotFoundError as e:
        return np.nan


def parse_runtime(l):
    pattern = "\s*hr:\s*(?P<hr>\d*)\s*min:\s*(?P<min>\d*)\s*sec:\s*(?P<s>\d*)\s*"
    m = re.match(pattern, l)
    if m is None:
        runtime=np.NaN
    else:
        runtime = compute_runtime_minutes(m.groupdict())
    return runtime


def compute_runtime_minutes(d):
    converter = {"hr": 60, "min": 1, "s": 1/60}
    return sum([int(val) * converter[key] for key, val in d.items()])
```

```python
futures = client.map(read_run_time, output_paths)
```

```python
runtimes = pd.Series(
    client.gather(futures),
    name="runtime minutes",
    index=pd.Index(model_run_ids, name="model_run_id")
).to_xarray().swap_dims(model_run_id="realization")
```

```python
sns.displot(runtimes, stat="percent", rug=True)
#plt.xscale("log")
```

## What are the parameters that determine how long the model runs?

```python
def process_multi_output_indices(problem_spec):
    return xr.concat(
        [df.to_xarray().rename(index="input_variable") for df in problem_spec.to_df()],
        dim=pd.Index(problem_spec["outputs"], name="output_variable")
    )
```

```python
# Only use the original parameters for the sensitivity analysis, not the transformed ones
# Also use the ones where I have generated samples previously
idata_params = ["log_tce_intrinsic_rate_constant", "log_ssa", "logit_initial_mineral_fraction"]
additional_params = pd.MultiIndex.from_product(
    [idata_params, idata.posterior.mineral.values]
).to_frame(index=False, name=["parameter", "index"])
untransformed_params = (
    pd.concat([priors[["parameter", "index"]], additional_params])
    .agg('/'.join, axis=1)
    .to_xarray()
    .swap_dims(index="parameter")
)
```

```python
diagnostics = xr.Dataset(
    {
        "finished": finished,
        "completed_simulation_time": qoi["simulated_time"],
        "runtime": runtimes.set_xindex("model_run_id").sel(model_run_id=finished.model_run_id),
    }
)
diagnostics = diagnostics.to_array("criterion").T
sp = sa.ProblemSpec(
    {
        "names": untransformed_params.values,
        "bounds": [(np.nan, np.nan) for param in untransformed_params.parameter],
        "outputs": diagnostics.criterion.values
    }
)

sp.set_samples(params_ds.sel(parameter=untransformed_params).values)
sp.set_results(diagnostics.values)
```

```python
diagnostics
```

```python
sp.analyze_pawn()
```

```python
indices = process_multi_output_indices(sp)
```

```python
fig, ax = plt.subplots(
    figsize=(6, 8),
    # gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin)
)
normalized = (indices["median"] - indices["median"].min("input_variable")) / (indices["median"].max("input_variable") - indices["median"].min("input_variable"))
sns.heatmap(normalized.to_pandas().T, yticklabels=True, ax=ax)
```

```python
sorted_indices = indices.sel(output_variable="finished").sortby("median", ascending=False)
problematic_params = sorted_indices.isel(input_variable=slice(None, 8)).input_variable
```

```python
df = xr.merge([params_ds.rename("parameter_value"), finished]).to_dataframe().drop(columns=["aqueous_reaction", "x", "model_run_id"])
```

```python
g = sns.displot(
    data=df.loc[problematic_params.values].reset_index(),
    x="parameter_value",
    hue="finished",
    rug=True,
    kind="hist",
    #stat="probability",
    col="parameter",
    col_wrap=4,
    facet_kws=dict(sharey=False, sharex=False),
    common_bins=False
)
```

```python
sns.displot(
    data=df.loc["log10_rate_constant/goethite_dirb"],
    x="parameter_value",
    hue="finished",
    rug=True,
    kind="hist",
    stat="probability"
)
```

Failing simulations occur at a broad range of DIRB rate constants.
However, the distributions of rate constants associated with successful and failing simulations do not overlap completely.
The distribution for failing simulations is somewhat shifted to larger rate constants.
Particularly at log10 rate constants of about 6, there are many failing and few successful simulations.
In general, however, the failure cannot be explained by the values of a single parameter.
It must be parameter combinations that lead to problematic results.

# Compute sensitivity indices

```python
quantities_of_interest = {
    "tce_flux_reduction": relative_cumulative_tce_flux,
    "integrated_dirb_rate": qoi["integrated_dirb_rate"],
    "integrated_sulfate_reduction_rate": qoi["integrated_sulfate_reduction_rate"],
    "average_mineral_content_change_FeS": qoi["average_mineral_content_change"].sel(mineral="FeS(am)", drop=True),
    "average_mineral_content_change_pyrite": qoi["average_mineral_content_change"].sel(mineral="Pyrite", drop=True),
    "average_mineral_content_change_siderite": qoi["average_mineral_content_change"].sel(mineral="Siderite", drop=True),
    "tce_removal_contributions_FeS": qoi["tce_removal_contributions"].sel(mineral_reaction="FeS(am)_tce_r", drop=True),
    "tce_removal_contributions_pyrite": qoi["tce_removal_contributions"].sel(mineral_reaction="Pyrite_tce_re", drop=True),
    "tce_removal_contributions_siderite": qoi["tce_removal_contributions"].sel(mineral_reaction="Siderite_tce_", drop=True),
}
```

```python
# Drop any realizations that did not finish und thus have NaNs in their output
# Remove datapoint with obvious nonsense as results
valid_outputs = xr.Dataset(quantities_of_interest).dropna(dim="realization")
valid_outputs = valid_outputs.pint.dequantify().to_array("output_variable").transpose("realization", "output_variable")
valid_inputs = (
    params_ds
    .sel(realization=valid_outputs.realization, parameter=untransformed_params)
)
```

```python
valid_inputs
valid_outputs
valid_inputs_pd=valid_inputs.to_dataframe()
valid_outputs_pd=valid_outputs.to_dataframe('valid_outputs_pd')
valid_inputs_pd.to_csv('valid_inputs',header=True,index=True)
valid_outputs_pd.to_csv('valid_outputs',header=True,index=True)
```

```python
qois_fluxes_raw = [compute_fluxes_raw(delayed) for delayed in with_concat_dim]
con_fluxes=dask.delayed(xr.concat)(qois_fluxes_raw, "realization")
q_fluxes=con_fluxes.compute()
q_fluxes
#fluxes_pd=fluxes.to_dataframe()
#fluxes_pd.to_csv('fluxes',header=True,index=True)
#first_dataset = with_concat_dim[0]
#origin_raw_data=first_dataset.to_dataframe()
#origin_raw_data.to_csv('origin_raw_data.csv',header=True,index=True)
```

```python
q_fluxes_pd=q_fluxes.to_dataframe()
q_fluxes_pd.to_csv('q_fluxes.csv',header=True,index=True)
```

```python
def make_problemspec():
    sp = sa.ProblemSpec(
        {
            "names": valid_inputs.parameter.values,
            "bounds": [(np.nan, np.nan) for param in valid_inputs.parameter],
            "outputs": valid_outputs.output_variable.values
        }
    )

    sp.set_samples(valid_inputs.values)
    sp.set_results(valid_outputs.values)
    return sp
```

# How big are correlations of parameters in the inputs?

1. Compute the correlation matrix

```python
corr = xr.DataArray(
    np.corrcoef(valid_inputs.T),
    dims=("parameter", "parameter_to"),
    coords={"parameter": valid_inputs.parameter, "parameter_to": valid_inputs.parameter.swap_dims(parameter="parameter_to")},
    name="correlation"
)
```

2. Check what the most negative correltion coefficient is.

```python
corr.min()
```

Find out which parameters correspond to that correlation coefficient

```python
idxs_min = np.argwhere(corr.data == corr.min().values)
corr.isel(parameter=idxs_min[0][0], parameter_to=idxs_min[0][1])
```

Check what the largest correlation coefficient is, not considering the main diagonal, where the correlation coefficient is one by construction.

```python
off_diagonal = ~np.eye(*corr.shape).astype(bool)
corr.where(off_diagonal, 0).max()
```

```python
idxs_max = np.argwhere(corr.data == corr.where(off_diagonal, 0).max().values)
corr.isel(parameter=idxs_max[0][0], parameter_to=idxs_max[0][1])
```

```python
corr.to_dataframe(name="correlation").unstack()
```

Plot the correlation matrix

```python
sns.heatmap(corr.to_dataframe().unstack().droplevel(0, axis=1), center=0, vmin=-1, vmax=1)
```

## High-dimensional model representation (HDMR)

```python
sp_hdmr = make_problemspec()
sp_hdmr.analyze_hdmr()
```

```python
indices_hdmr = process_multi_output_indices(sp_hdmr)
```

The analysis provides three indices for each parameter and parameter combination $p_j$.
$S_{p_j}$ indicates the total contribution of $p_j$ which is composed of its structural contribution $S_{p_j}^a$ and its correlative contribution $S_{p_j}^b$. The correlative contribution can also be negative.

```python
indices_hdmr.isel(output_variable=0).sortby("ST", ascending=False).to_pandas().dropna()
```

```python
ST = indices_hdmr["ST"].dropna("input_variable").where(lambda x:  (x < 100) & (x > -100)).to_pandas()
```

```python
sns.clustermap(ST.dropna().T, col_cluster=False)
```

The total sensitivity indicies $S_{T_i}$ is the sum of all total contributions: $S_{T_i} = \sum_j S_{ij}$.
They can be considered reliable if the sum of the $S_{p_j}$ over all parameters, $\sum_{j=1}^{n_p}S_{p_j}\approx 1$.
Check if that is the case:

```python
indices_hdmr["S"].sum("input_variable")
```

In theory, $S_{p_j}$ should equal the sum of $S_{p_j}^a$ and $S_{p_j}^b$. Check if that is the case:

```python
S = indices_hdmr.isel(output_variable=0)["S"].to_series()
sum_Sa_Sb = indices_hdmr.isel(output_variable=0).to_dataframe().eval("Sa + Sb").rename("Sa + Sb")

pd.concat([S, sum_Sa_Sb], axis=1)
```

## Delta Moment-Independent Measure

- This method is based on the probability density of output variables.
- The sensitivity index measures the expected difference between the unconditional output distribution and the distribution that is obtained when one input parameter is fixed. The difference is measured by the area between the curves. The expecteation is applied because the parameter that is fixed is itself a random variable.
- When $\delta_i$ is 0, the output is not sensitive to parameter $i$ at all.
- The method can be applied to correlated inputs.

```python
sp_dmim = make_problemspec()
sp_dmim.analyze_delta()
```

```python
indices_dmim = process_multi_output_indices(sp_dmim)
```

```python
indices_dmim.isel(output_variable=0).to_pandas().sort_values(by="delta", ascending=False)
```

```python
sns.heatmap(indices_dmim["delta"].to_pandas().T)
```

## PAWN

- This is a moment-independent method for sensitivity analysis.
- The idea is similar as for DMIM.
  However, the sensitivity index is based on differences of CDFs instead of PDFs. To measure the difference, a Kolmogorov-Smirnov statistics is used. 

```python
sp_pawn = make_problemspec()
sp_pawn.analyze_pawn()
```

```python
indices_pawn = process_multi_output_indices(sp_pawn)
```

```python
indices_pawn.isel(output_variable=0).to_pandas().sort_values("median", ascending=False)
```

```python
sns.heatmap(indices_pawn["median"].to_pandas().T)
```

## RBD_FAST

Careful: This method approximates variance-based sensitivity indices.
It is based on the assumption that input parameters are *uncorrelated*, which is not the case for my model.

```python
sp_fast = make_problemspec()
sp_fast.analyze_rbd_fast()
```

```python
indices_fast = process_multi_output_indices(sp_fast)
```

```python
indices_fast.isel(output_variable=0).to_pandas().sort_values(by="S1", ascending=False)
```

```python
sns.heatmap(indices_fast["S1"].to_pandas().T)
```
