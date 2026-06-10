---
title: '3D Transport-based Morphometry (3D-TBM) for medical image analysis'

tags:
  - Medical Image Analysis
  - Transport-Based Morphometry
  - Optimal Transport
  - 3D Brain Image

authors:
  - name: Hongyu Kan
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Kristofor Pas
    affiliation: 1
  - name: Ivan Medri
    affiliation: 1
  - name: Naqib Sad Pathan
    affiliation: 1
  - name: Natasha Ironside
    affiliation: 1
  - name: Shinjini Kundu
    affiliation: 2
  - name: Gustavo Kunde Rohde
    affiliation: 1

affiliations:
  - name: University of Virginia
    index: 1
  - name: Washington University in St. Louis 
    index: 2

date: 10 June 2026
bibliography: paper.bib


---

# Summary

Transport-Based Morphometry (TBM) has emerged as a new framework for extracting physiologically interpretable information from 3D medical images. By embedding images into a transport domain via invertible transformations, TBM facilitates effective classification, regression, and other tasks using transport-domain features. Crucially, the inverse mapping enables the projection of analytic results back into the original image space, allowing researchers to directly interpret clinical features associated with model outputs in a spatially meaningful way. To facilitate broader adoption of TBM in clinical imaging research, we present 3D-TBM, an open source python-based tool designed for morphological analysis of 3D medical images. The framework includes data preprocessing, computation of optimal transport embeddings, and analytical methods such as visualization of main transport directions, together with techniques for discerning discriminating directions and related analysis methods. We also provide comprehensive documentation and practical tutorials to support researchers interested in applying 3D-TBM in their own medical imaging studies. The source code is publicly available through PyTransKit [@pytranskit2023].


# Statement of Need

Three-dimensional (3D) transport-based morphometry (TBM) [@kundu2018discovery; @wang2013linear] has recently emerged as a promising framework for morphological analysis in medical imaging, with applications including the prediction of hematoma expansion[@ironside2025predictive], the study of brain tissue distribution changes associated with cardiorespiratory fitness[@kundu2021investigating], the identification of autism-related endophenotypes[@kundu2024discovering], and among others[@kundu2020enabling; @kundu2019assessing]. The key idea is based on the mathematical principles of optimal mass transport [@villani2008optimal], which is used to embed medical images into a novel mathematical space called the Linear Optimal Transport (LOT) [@wang2013linear] space. In the transport domain, morphological variations can be captured and analyzed in a more direct and linear manner [@kundu2024discovering]. Moreover, since the optimal transport transformation is invertible, the analysis results can be 'mapped back' to the original image space. This property not only enables classification and regression tasks, but also allows visualization of anatomical regions in the original images that are associated with the model outputs, thus providing clinical interpretability.

Despite the significant advantages of 3D TBM in medical image analysis there is currently a lack of user-friendly software readily available for clinical researchers. With the growing interest in optimal transport within clinical research, and medical imaging in particular, an increasing number of researchers are facing difficulties due to the lack of suitable tools. Therefore, providing a user-friendly and accessible framework for 3D TBM has become particularly important.

# State of the field

Although other medical imaging tools offer convenient functionalities for morphological analysis - such as registration, segmentation, and even neural network training - they do not provide tools to perform transport-based morphometry. Advanced Normalization Tools (ANTs) is a C++-based command-line library for high-dimensional biomedical image registration and analysis, enabling the statistical exploration, visualization, and integration of large-scale imaging data across modalities, species, and organ systems \citep{avants2009advanced}. Extensions of ANTs include ANTsR and ANTsPy, as well as ANTsX, which integrates deep learning methods into the ANTs framework \citep{tustison2021antsx,tustison2024antsx}. MONAI is an open-source, community-driven PyTorch-based framework that extends PyTorch for medical imaging, providing specialized architectures, transformations, and utilities to facilitate the development and deployment of healthcare AI models \citep{cardoso2022monai}.
FreeSurfer is a comprehensive neuroimaging analysis suite that quantifies functional, connectional, and structural brain properties, evolving from cortical surface modeling to the automated reconstruction of most macroscopically visible brain structures from T1-weighted images \citep{fischl2012freesurfer,dale1999cortical}. SPM is a software package for analyzing brain imaging data sequences, including cross-sectional datasets from multiple cohorts and longitudinal time-series from individual subjects \citep{friston1994statistical,penny2011statistical}. AFNI (Analysis of Functional NeuroImages) is a comprehensive software suite composed of C, Python, and R programs, along with shell scripts, designed primarily for analyzing and visualizing various MRI modalities, including anatomical, functional (fMRI), and diffusion-weighted (DW) data\citep{cox1996afni,cox1997software}. However, none of these software packages currently provide dedicated tools for TBM analysis.

# Software design

The algorithm is based on the work of \citep{kundu2018discovery} and provides a complete pipeline covering data preprocessing, computing the LOT embedding \citep{wang2013linear, kolouri2017optimal, basu2014detecting}, and data analysis. The corresponding components include preprocessing for the calculation of LOT, the LOT computation process \citep{kundu2018discovery}, principal component analysis (PCA) \citep{jolliffe2016principal}, linear discriminant analysis (LDA/PLDA) \citep{balakrishnama1998linear,wang2011penalized}, canonical correlation analysis (CCA) \citep{hardoon2004canonical}, and visualization of analytical results. Figure \ref{fig:placeholder} illustrates the workflow of 3D-TBM for LOT embedding and model-based visualization.



% In this section, we present the 3D-TBM in detail, a Python-based tool that encapsulates complex mathematical logic and computations into simple interfaces, with the aim of providing researchers interested in TBM with a convenient and accessible framework.

TBM assumes that the data to be mined for information has been pre-processed accordingly (Step 1). For brain images, for example, this assumes that all brains have been segmented (i.e. skull removed), have the same resolution, and are roughly aligned to the same atlas/coordinate system. Denote this 3D medical image dataset as $I_1,...,I_N: R^{h\times w \times d} \to R^+$. Furthermore, denote a reference (in medical imaging typically an atlas) image as $I_0:R^{h\times w \times d} \to R^+$, where $h,w, d$ is the dimension of the 3D medical images and $N$ is the number of 3D medical images. Futhermore, TBM assumes that all images are normalized according to $\sum_x I_i(\textbf{x}) = 1, \ i= 0,...,N$.

% \begin{equation}
%     \sum_x I_i(\textbf{x}) = 1, \ i= 0,...,N
% \label{eq:noraml}
% \end{equation}

Step 2 in the TBM pipeline is to compute the linearized optimal transport (LOT) embedding for each image. This is done by solving the following mathematical minimization problem \citep{kundu2018discovery}:
\begin{equation}
    \begin{aligned}
    f^*_i(\textbf{x}) = \arg \min_{f_i} \int |\textbf{x}-f_i(\textbf{x})|^2I_0(\textbf{x})d\textbf{x},\\ s.t.\ det(Df_i(\textbf{x}))
    I_i(f_i(\textbf{x})) = I_0(\textbf{x})
    \end{aligned}
\end{equation}
where $D$ denotes the Jacobian matrix, $f_i : R^3 \to R^3$ is a mass preserving mapping from $I_0$ to $I_i$. Since the transport map is invertible, it can also be regarded as a representation (or embedding) for image $I_i$.


The features (denoted as $f_i$) obtained from the LOT embedding can be utilized for model-based data analysis, including, but not limited to, dimensionality reduction, classification, and regression(Step 3). Correspondingly, the principal components in PCA or the discriminant direction learned by PLDA can be mapped back to the original feature space, which are the optimal transport map features. According to Equation \ref{eq:inverse}, the optimal transport map of an image can be inverted from the transport domain back to the image domain, thus reconstructing the 3D image(Step 4). This allows us to directly examine the learned data representations in the image domain, thereby enhancing the interpretability of the models:

\begin{equation}
\label{eq:inverse}
    I_{recon}(\mathbf{x}) =  (D_{f^{-1}}(\mathbf{x})'I_0( f^{-1}(\mathbf{x}))
\end{equation}
where $f^{-1}$ stands for the inverse of deformation field $f$ and $D_{f^{-1}}(\mathbf{x})$ stands for the Jacobian determinant of the inverse deformation field at $\mathbf{x}$.

# Research impact statement


# AI usage disclosure



# Acknowledgements
Authors gratefully acknowledge funding from the ONR (N000142212505) and the NIH (GM130825, U54-CA274499) in contributing to a portion of this work. 
Authors also acknowledge the source of the IXI data: https://brain-development.org/ixi-dataset/ .
# References
