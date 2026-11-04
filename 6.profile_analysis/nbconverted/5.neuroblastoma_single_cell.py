#!/usr/bin/env python
# coding: utf-8

# # Neuroblastoma single-cell UMAP with cluster examples
# 
# Computes a UMAP of the single cells of the neuroblastoma cell lines, groups the cells into clusters, and finds representative cells of each cluster to show as images.
# 
# **Why:** neuroblastoma is the cancer group with the most cell lines (12).
# Showing what the clusters of its UMAP look like, and which cell lines they come from, shows how the cell lines differ at the level of single cells.
# 
# **Input:** `<plate>_sc_feature_selected.parquet` of the plates that have neuroblastoma cell lines (multivariate).
# Each plate has its own features, so the UMAP uses the features that these plates share.
# The profiles of each plate are standardized to that plate.
# 
# **Method:** it is the method of `3.single_cell_embedding.ipynb` for one cancer group, with more cells.
# At most `cells_per_line_per_plate` cells of each cell line on each plate are drawn at random (fixed seed), all time points are used, and the cells are clustered with k-means on the UMAP coordinates.
# The representative cells of a cluster are the `examples_per_cluster` cells closest to its center.
# Their images are cut from the raw images with `utils/cell_image_utils.py` (raw images, no illumination correction, each channel stretched in its own crop).
# 
# **Output:** in `results/`
# - `neuroblastoma_umap.parquet`: one row per sampled cell with the cell line, plate, well, site (field number), nucleus position in the image, time point, UMAP coordinates, and cluster.
# - `neuroblastoma_cluster_composition.csv`: the number and the share of the cells of each cluster that come from each cell line.
# - `neuroblastoma_examples.csv` and `neuroblastoma_examples/`: the representative cells and their images (`cluster<n>_example<m>.png`).
# - `neuroblastoma_umap.png`: the UMAP colored by cluster and by cell line.

# ## Import libraries

# In[1]:


import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
import yaml
from cell_image_utils import crop_cell, find_image_dir
from sklearn.cluster import KMeans

# UMAP warns that a fixed random seed runs it on one thread, which is what keeps the embedding reproducible
warnings.filterwarnings("ignore", message="n_jobs value")


# ## Set paths and parameters

# In[2]:


metadata_path = pathlib.Path("../cell_lines/pccma_cell_line_metadata.csv")
groups_path = pathlib.Path("../cell_lines/plotting_helpers/cancer_type_categories.yaml")

# single-cell feature-selected profiles and raw images: use the local data folder and fall back to the bandicoot mount
local_dir = pathlib.Path("../4.preprocess_features/data/single_cell_profiles")
bandicoot_root = pathlib.Path("~/mnt/bandicoot/PCCMA_data").expanduser()
profile_dir = local_dir if any(local_dir.glob("BR*_sc_feature_selected.parquet")) else bandicoot_root / "Atlas_v2_profiles/single_cell_profiles"
screen_dir = bandicoot_root / "Atlas_v2_screen"

results_dir = pathlib.Path("./results")
(results_dir / "neuroblastoma_examples").mkdir(parents=True, exist_ok=True)

cancer_group = "Neuroblastoma"
cells_per_line_per_plate = 2000  # cells drawn per cell line on each plate
n_clusters = 8
examples_per_cluster = 4  # representative cells shown per cluster
crop_size = 160  # pixels of an example image (about 95 um)
umap_parameters = {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "metric": "euclidean", "random_state": 0}
rng_seed = 0


# ## Helper function

# In[3]:


def load_cell_line_groups(metadata_path, groups_path):
    """One row per cell line with its cancer type, cancer group and tissue site.

    The cancer groups and tissue sites come from cancer_type_categories.yaml, which lists
    which `cancer_type` values of the metadata belong to each group.
    """
    metadata = pd.read_csv(metadata_path).drop_duplicates("cell_line")  # U2-OS has one row per plate condition
    categories = yaml.safe_load(open(groups_path))
    groups = pd.DataFrame(
        [
            (cancer_type, group["name"], site["name"])
            for site in categories["sites"]
            for group in site["groups"]
            for cancer_type in group["cancer_types"]
        ],
        columns=["cancer_type", "cancer_group", "tissue_site"],
    )
    return metadata[["cell_line", "cancer_type", "origin"]].merge(groups, on="cancer_type", how="left")


# ## Neuroblastoma cell lines, their plates, and the shared features

# In[4]:


neuroblastoma_lines = load_cell_line_groups(metadata_path, groups_path).query("cancer_group == @cancer_group")["cell_line"]

plate_files = sorted(profile_dir.glob("BR*_sc_feature_selected.parquet"))
plate_files = [
    file for file in plate_files
    if pd.read_parquet(file, columns=["Metadata_cell_line"])["Metadata_cell_line"].isin(neuroblastoma_lines).any()
]


def numeric_features(schema):
    return {
        field.name for field in schema
        if not field.name.startswith(("Metadata_", "Image_")) and (pa.types.is_floating(field.type) or pa.types.is_integer(field.type))
    }


features = sorted(set.intersection(*[numeric_features(pq.read_schema(file)) for file in plate_files]))
print(f"{len(neuroblastoma_lines)} cell lines on {len(plate_files)} plates, {len(features)} features shared by these plates")


# ## Subsample the neuroblastoma cells

# In[5]:


cell_columns = [
    "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_cell_line", "Metadata_time_point",
    "Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y",
]

rng = np.random.default_rng(rng_seed)
sampled = []
for plate_file in plate_files:
    plate_df = pd.read_parquet(plate_file, columns=cell_columns + features)
    plate_df = plate_df[plate_df["Metadata_cell_line"].isin(neuroblastoma_lines)]
    chosen = [
        rng.choice(np.asarray(rows), size=min(len(rows), cells_per_line_per_plate), replace=False)
        for rows in plate_df.groupby("Metadata_cell_line").indices.values()
    ]
    sampled.append(plate_df.iloc[np.concatenate(chosen)])
    print(f"{plate_file.name}: {len(sampled[-1])} of {len(plate_df)} neuroblastoma cells")

cells_df = pd.concat(sampled, ignore_index=True)
print(f"{len(cells_df)} cells from {cells_df['Metadata_cell_line'].nunique()} cell lines")


# ## UMAP and clusters

# In[6]:


embedding = umap.UMAP(**umap_parameters).fit_transform(cells_df[features].to_numpy())
cells_df["umap_1"], cells_df["umap_2"] = embedding[:, 0], embedding[:, 1]

kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=rng_seed).fit(embedding)
cells_df["cluster"] = kmeans.labels_ + 1

keep = ["Metadata_cell_line", "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_time_point",
        "Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y", "umap_1", "umap_2", "cluster"]
cells_df[keep].to_parquet(results_dir / "neuroblastoma_umap.parquet", compression="zstd")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for cluster, cluster_df in cells_df.groupby("cluster"):
    axes[0].scatter(cluster_df["umap_1"], cluster_df["umap_2"], s=1.5, alpha=0.6, label=cluster)
    axes[0].annotate(cluster, kmeans.cluster_centers_[cluster - 1], fontsize=12, weight="bold", ha="center")
for color, (line, line_df) in zip(plt.get_cmap("tab20").colors, cells_df.groupby("Metadata_cell_line")):
    axes[1].scatter(line_df["umap_1"], line_df["umap_2"], s=1.5, alpha=0.6, color=color, label=line)
axes[0].set(title="Clusters", xlabel="UMAP 1", ylabel="UMAP 2")
axes[1].set(title="Cell lines", xlabel="UMAP 1")
axes[1].legend(markerscale=6, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(results_dir / "neuroblastoma_umap.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ## Which cell lines make up each cluster
# 
# The number of cells of each cluster from each cell line, and the share of the cluster.

# In[7]:


counts = cells_df.groupby(["cluster", "Metadata_cell_line"]).size().rename("n_cells").reset_index()
counts["share_of_cluster"] = counts["n_cells"] / counts.groupby("cluster")["n_cells"].transform("sum")
counts = counts.rename(columns={"Metadata_cell_line": "cell_line"})
counts.to_csv(results_dir / "neuroblastoma_cluster_composition.csv", index=False)

print("Cell line with the largest share of each cluster:")
largest = counts.loc[counts.groupby("cluster")["share_of_cluster"].idxmax()]
print(largest.assign(cells_in_cluster=largest["cluster"].map(cells_df["cluster"].value_counts())).round(2).to_string(index=False))


# ## Representative cell images
# 
# The cells closest to the center of each cluster, cut from the raw images of their plate.

# In[8]:


example_rows = []
for cluster in range(1, n_clusters + 1):
    in_cluster = np.where(cells_df["cluster"].to_numpy() == cluster)[0]
    distance = ((embedding[in_cluster] - kmeans.cluster_centers_[cluster - 1]) ** 2).sum(axis=1)
    for number, row in enumerate(in_cluster[np.argsort(distance)[:examples_per_cluster]], start=1):
        cell = cells_df.iloc[row]
        image = crop_cell(
            find_image_dir(screen_dir, cell["Metadata_Plate"]), cell["Metadata_Well"], cell["Metadata_Site"],
            cell["Metadata_Nuclei_Location_Center_X"], cell["Metadata_Nuclei_Location_Center_Y"], size=crop_size,
        )
        image_file = f"cluster{cluster}_example{number}.png"
        plt.imsave(results_dir / "neuroblastoma_examples" / image_file, np.clip(image, 0, 1))
        example_rows.append({**cell[keep].to_dict(), "example": number, "image_file": f"neuroblastoma_examples/{image_file}"})

pd.DataFrame(example_rows).to_csv(results_dir / "neuroblastoma_examples.csv", index=False)
print(f"{len(example_rows)} example cell images saved")

