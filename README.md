# CM3070 Thesis Repository Collection

Final model deployed: https://storage.googleapis.com/davidcardozo-final-london-demo/index.html

**Note to the reviewer**: Please click (Wake Server) so that a gpu machine is provided to you (2 - 3 mins).

This repository is the top-level entrypoint for the CM3070 thesis project on curated breast imaging with CBIS-DDSM. It does not hold a single implementation. Instead, it organizes the full project as a set of Git submodules, each covering one stage of the workflow:

- dataset download
- DICOM to PNG conversion
- dataset inspection and visualization
- exploratory data analysis
- model training and evaluation
- model serving

The intent of this repository is reproducibility: one place to clone, initialize, and navigate the full thesis codebase.

## Author

David Cardozo  
Student, University of London  

## Thesis Objective

This thesis is motivated by the need for a scalable and computationally feasible pipeline for breast cancer image analysis in Colombia. More specifically, the project brings together data acquisition, preprocessing, validation, exploratory analysis, model development, evaluation, and deployment-oriented serving in order to support the construction of an end-to-end diagnostic workflow based on curated mammographic data. The aim is not merely to train isolated models, but to establish a reproducible technical pipeline through which breast cancer classification can be studied, compared, and operationalized under realistic computational constraints. An additional objective is to explore the practical capabilities of JAX, Flax, and TPU-based training in the context of medical imaging, both as a means of accelerating experimentation and as a way of assessing their suitability for large-scale deep learning workflows.

## Research Questions

This thesis is guided by the following research questions:

1. What level of performance can be achieved using classical linear baselines on the curated CBIS-DDSM patch classification task?
2. How much improvement is obtained when replacing linear baselines with deep learning models trained from scratch?
3. Do pretrained `timm` models provide a further performance advantage over from-scratch deep learning approaches?

## Project Contributions

The project makes the following practical contributions:

1. It modernizes the technical stack used to acquire and process the dataset, replacing ad hoc or informal acquisition practices with a more reproducible workflow.
2. It provides a correct and efficient pathway for downloading the source data directly, rather than relying on repackaged copies from secondary platforms such as Kaggle.
3. It establishes a scalable pipeline for converting the raw imaging files into formats that can be consumed consistently by downstream machine learning workflows.
4. It organizes the full process of downloading, converting, validating, exploring, modelling, evaluating, and serving the data into a single repository collection designed for reproducibility.

## Reproducibility Boundaries

This repository collection is designed to maximize reproducibility, but full reproduction of all stages depends on the computational environment available to the reader.

- The repository structure, submodule relationships, documentation, figure-generation code, and evaluation pipeline are reproducible directly from version-controlled sources.
- Python-based components are reproducible through the `uv`-managed environments defined in their respective repositories, and the JAX-based parts of the stack are included precisely to support a modern and reproducible research workflow.
- This project uses Google Cloud Platform (GCP) for its cloud-based stages. Because these stages are defined through version-controlled configuration files and containerized workloads, they are designed to be reproducible within GCP, subject to the availability of the required credentials, services, and compute resources.
- Many stages of the project are containerized, which makes them portable across execution environments. In practice, this means that substantial parts of the pipeline can be run locally, on hyperscaler infrastructure, or in other cloud environments, provided the required dependencies and resources are available.
- Data acquisition and preprocessing workflows are therefore reproducible in principle across environments, although some stages in this specific implementation also depend on Google Cloud services or access to persisted storage locations.
- Large-scale conversion and orchestration workflows in the current project configuration depend on Google Cloud Batch, Cloud Storage, Artifact Registry, and the corresponding credentials and quotas.
- TPU-based experiments require access to compatible TPU infrastructure, while GPU-based training and model serving require suitable accelerator hardware.
- For these reasons, the collection should be understood as fully reproducible at the level of code, configuration, and workflow design, while full execution of every stage may still depend on external infrastructure and resource availability.

## Repository role

This repo acts as a meta-repository for the thesis. The actual implementation work lives inside submodules.

At the root level, the project contains:

- `CM3070-Downloader` for acquiring CBIS-DDSM data
- `CM3070-Converter` for cloud-based DICOM to PNG conversion and TFDS preparation
- `CM3070-Visualizer` for dataset validation and visual inspection with TFDV / Facets
- `CM3070-EDA` for building a persistent FiftyOne dataset and computing embeddings/visualizations
- `CM3070-Models-Implementation-Evaluation` for figure generation, report artifacts, and model comparison
- `CM3070-Model-Serving-Triton-ONNX` for ONNX deployment with Triton on Cloud Run GPU

Inside `CM3070-Models-Implementation-Evaluation`, there are additional nested submodules:

- `linear-models`
- `models-training-with-tpus`
- `models-with-gradcam-curated-breast-imaging-ddsm`

These nested repositories support the implementation and evaluation layer of the thesis.

## Repository structure

```text
CM3070-curated_breast_imaging_ddsm/
├── CM3070-Downloader/
├── CM3070-Converter/
├── CM3070-Visualizer/
├── CM3070-EDA/
├── CM3070-Model-Serving-Triton-ONNX/
└── CM3070-Models-Implementation-Evaluation/
    ├── linear-models/
    ├── models-training-with-tpus/
    └── models-with-gradcam-curated-breast-imaging-ddsm/
```

## End-to-end workflow

The project is organized roughly in this order:

### 1. Download the raw dataset

Use `CM3070-Downloader` to retrieve CBIS-DDSM data. The repository packages the NBIA Data Retriever CLI in Docker, includes a bundled manifest for the dataset, and also contains Google Cloud build/batch configuration files.

Typical responsibility:

- build the downloader image
- run the NBIA retriever against the provided manifest
- persist downloaded files to a mounted host directory
- build the downloader container image with `cloudbuild.yaml`
- use `download.yaml` to rehydrate or resume the dataset state from the persisted Google Cloud Storage copy when running the Google Cloud workflow

In GCP terms, the two YAML files play different roles:

- `cloudbuild.yaml` is the image build step. It builds the downloader container from the repo's `Dockerfile` and pushes that image to Artifact Registry so it can be executed consistently in cloud infrastructure.
- `download.yaml` is the batch execution step. It defines the compute resources, attached disk, mounted volume, logging policy, and container command for a Google Cloud Batch job.

That separation is important:

- Cloud Build answers "how do I package the downloader environment?"
- Cloud Batch answers "where and with what machine do I run that packaged environment?"

In its current form, `download.yaml` is configured to launch a batch VM with an attached SSD work disk and synchronize an existing dataset copy from Google Cloud Storage onto that mounted disk. This should be understood as an operational safeguard rather than as a separate data source. In practical terms, the Google Cloud Storage bucket functions as a persisted intermediate copy of the downloaded dataset, which reduces repeated transfers from the upstream NBIA servers, helps avoid unnecessary load on that external service, and allows failed or repeated jobs to resume from an already populated dataset state. This design is consistent with the downloader container itself, whose Go-based retrieval workflow is intended to avoid re-downloading files that are already present.

### 2. Convert raw DICOM data into model-ready assets

Use `CM3070-Converter` for the cloud conversion pipeline. This repository is centered on Google Cloud Batch and Cloud Storage.

Its role includes:

- distributed DICOM to PNG conversion
- archiving converted images
- preparing TFDS / TFRecord-ready outputs
- building the worker container image for the conversion jobs

This stage is particularly well suited to large-scale parallel execution because the conversion of one image is, for practical purposes, independent of the conversion of the others. The repository therefore treats DICOM-to-PNG conversion as an embarrassingly parallel workload and uses Google Cloud Batch to distribute file conversion across many workers with minimal coordination overhead. In effect, this design substantially accelerates preprocessing by allowing the dataset to be partitioned into shards that can be processed concurrently rather than sequentially on a single machine.

### 3. Validate and visualize the dataset

Use `CM3070-Visualizer` to inspect dataset statistics with TensorFlow Data Validation.

This stage supports:

- sampling TFDS splits
- generating train / validation / test statistics
- inferring a schema
- exporting HTML visualizations for inspection

From a methodological perspective, this stage provides an explicit data validation layer before model development. Rather than treating the prepared dataset as correct by assumption, the visualizer makes it possible to inspect split-level statistics, verify basic structural properties of the samples, and identify potential irregularities in dimensions, channels, labels, or feature distributions. This is useful both for quality assurance and for documenting that the downstream modelling work is grounded in a dataset whose basic characteristics have been examined systematically.

### 4. Perform exploratory data analysis

Use `CM3070-EDA` to create a persistent FiftyOne dataset from the TFDS data and compute embedding-based visualizations.

This stage supports:

- building a local FiftyOne dataset
- exporting datasets for inspection
- computing embeddings
- running UMAP / t-SNE / PCA visualizations
- similarity-based browsing in the FiftyOne app

Whereas the previous stage is primarily concerned with validation and summary statistics, this stage is oriented toward exploratory interpretation. By constructing a persistent FiftyOne dataset and attaching learned embeddings plus low-dimensional projections, the repository supports qualitative inspection of structure within the dataset, including clustering behaviour, class separation, and the presence of visually similar samples across splits. In the current implementation, embeddings are computed for a selected FiftyOne Model Zoo backbone, and the resulting representation can then be projected with UMAP, t-SNE, or PCA. Accordingly, the repository should be understood as supporting configurable embedding-based exploration rather than exhaustively computing every possible embedding or projection. In the context of the thesis, this provides a useful bridge between raw data preparation and formal modelling by enabling closer examination of how the curated dataset is organized in feature space.

### 5. Train models and reproduce thesis figures

Use `CM3070-Models-Implementation-Evaluation` as the report and experiment integration repository.

This repository ties together:

- `linear-models` for classical baselines
- `models-training-with-tpus` for JAX/Flax TPU experiments
- `models-with-gradcam-curated-breast-imaging-ddsm` for PyTorch/timm experiments with Grad-CAM
- `reports/` and `scripts/` for regenerated figures, extracted metrics, and report assets

According to the current project documentation, this layer is used to regenerate the thesis figures and compare:

- linear baselines
- models trained from scratch on TPUs
- pretrained models fine-tuned with Grad-CAM support

Conceptually, this stage is where the separate modelling lines are brought into comparative relation. The repository does not represent a single training pipeline; rather, it functions as the integration layer in which heterogeneous experiments are assembled, their outputs are aligned, and their results are translated into report-ready tables and figures. This is especially important in the present thesis because the compared approaches differ both methodologically and computationally, ranging from classical linear baselines to TPU-based training and pretrained GPU fine-tuning. The evaluation repository therefore serves as the point at which those distinct strands are rendered comparable within a single narrative and empirical framework.

### 6. Serve the selected ONNX model

Use `CM3070-Model-Serving-Triton-ONNX` for deployment.

This repository contains the serving stack for:

- NVIDIA Triton Inference Server
- Nginx reverse proxy
- Cloud Build deployment flow
- Google Cloud Run with GPU support

The selected deployment model documented in the evaluation repository is the ONNX-exported ResNet50 baseline.

From the standpoint of the overall project, this stage narrows the thesis from comparative experimentation to deployable inference. Rather than attempting to operationalize every trained model, the serving repository focuses on a selected model that offers a suitable trade-off between predictive quality and deployment practicality. The implementation couples Triton, Nginx, Cloud Build, and Cloud Run GPU infrastructure into a reproducible serving stack, thereby extending the thesis beyond offline evaluation and into an executable inference deployment scenario.

## Quick start

Clone the meta-repo and initialize all submodules, including nested ones:

```bash
git clone git@github.com:Davidnet/CM3070-Thesis-University-curated_breast_imaging_ddsm.git
cd CM3070-Thesis-University-curated_breast_imaging_ddsm
git submodule update --init --recursive
```

To update all submodules later:

```bash
git submodule update --remote --recursive
```

## Working with nested submodules

One important detail in this project is that `CM3070-Models-Implementation-Evaluation` is itself a Git repository with its own nested submodules. Because of that:

- clone with `--recursive`, or run `git submodule update --init --recursive`
- if you change a nested model repository, commit and push it first
- then commit the updated submodule pointer in `CM3070-Models-Implementation-Evaluation`
- finally commit the updated pointer in this top-level thesis repo if needed

That order matters for keeping all referenced commits available on their remotes.

## Suggested usage order

If you are approaching the project from scratch, a reasonable order is:

1. `CM3070-Downloader`
2. `CM3070-Converter`
3. `CM3070-Visualizer`
4. `CM3070-EDA`
5. `CM3070-Models-Implementation-Evaluation`
6. `CM3070-Model-Serving-Triton-ONNX`

## Notes

- Most repositories are independent and maintain their own environments and dependencies.
- Cloud-oriented components assume access to Google Cloud services and the relevant credentials.
- Every Python repository in this collection uses `uv` as its package and environment manager.
- The project spans local workflows, Google Cloud Batch pipelines, TPU training, and GPU-backed serving, so prerequisites differ by submodule.

## Submodule summary

| Repository | Purpose |
|---|---|
| `CM3070-Downloader` | Retrieve CBIS-DDSM data using the NBIA Data Retriever CLI in Docker |
| `CM3070-Converter` | Convert DICOM to PNG and prepare TFDS / TFRecord assets on Google Cloud |
| `CM3070-Visualizer` | Validate TFDS dataset statistics with TFDV / Facets |
| `CM3070-EDA` | Build a persistent FiftyOne dataset and compute visual embeddings |
| `CM3070-Models-Implementation-Evaluation` | Central evaluation repo for report figures, metrics, and experiment integration |
| `CM3070-Model-Serving-Triton-ONNX` | Serve the selected ONNX model with Triton on Cloud Run GPU |
| `linear-models` | Classical linear / feature-engineered baselines |
| `models-training-with-tpus` | TPU-based JAX/Flax deep learning experiments |
| `models-with-gradcam-curated-breast-imaging-ddsm` | PyTorch/timm models with Grad-CAM support |

## Thesis scope

At a high level, this thesis repository collection covers:

- curation and preparation of the CBIS-DDSM-based dataset
- inspection and exploratory analysis of the prepared data
- comparison of classical and deep learning approaches
- generation of figures and report artifacts
- deployment of the selected model for inference serving

For implementation details, commands, and environment setup, consult the README in each submodule.
