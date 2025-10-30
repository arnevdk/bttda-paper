# BTTDA: Block-Term Tensor Discriminant Analysis for Brain-Computer Interfacing

## Abstract

Brain-computer interfaces (BCIs) allow for direct communication between the brain and external devices, frequently using electroencephalography (EEG) to record neural activity. Dimensionality reduction and structured regularization are essential for effectively classifying task-related brain signals, including event-related potentials (ERPs) and motor imagery (MI) rhythms. Current tensor-based approaches, such as Tucker and PARAFAC decompositions, often lack the flexibility needed to fully capture the complexity of EEG data. This study introduces Block-Term Tensor Discriminant Analysis (BTTDA): a novel tensor-based and supervised feature extraction method designed to enhance classification accuracy by providing flexible multilinear dimensionality reduction. Extending Higher Order Discriminant Analysis (HODA), BTTDA uses a novel and interpretable forward model for HODA combined with a deflation scheme to iteratively extract relevant block terms, improving feature representation for classification. BTTDA and a rank-1 variant PARAFACDA were evaluated on publicly available ERP (second-order tensor) and MI (third-order tensor) EEG datasets. Benchmarking in the MOABB framework revealed that BTTDA and PARAFACDA significantly outperform the traditional HODA method in ERPdatasets, resulting in state-of-the art decoding performance. For MI, decoding results of HODA, BTTDA and PARAFACDA were subpar, but BTTDA still significantly outperformed HODA. The block-term structure of BTTDA enables interpretable and more efficient dimensionality reduction without compromising discriminative power. This offers a promising and adaptable approach for feature extraction in BCI and broader neuroimaging applications.

## Keywords

tensor discriminant analysis,
brain-computer interface,
block-term decomposition,
multilinear decoding,
event-related potentials,
motor imagery
