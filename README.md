# DS Project 3: Online Advertisement Visual Analysis

#### Authors: Diya Gupta, Delaney Brown, Connor Powell (Leader)

## Overview  
This repository contains datasets, scripts, and outputs for our third project for DS 4002, which focuses on analyzing and identifying the most common structural and visual elements in online advertisements. The dataset used is sourced from the Hugging Face repository **AdImageNet** by PeterBrendan, which includes advertisement images and associated metadata stored in Parquet format [2].  

Due to GitHub size limitations, the DATA folder is empty, but the dataset is accessible through this [Google Drive link](https://drive.google.com/drive/folders/1NHBysZ5VgMlqOjqfnfftpxTf-JamfPjm?usp=drive_link).  
The SCRIPTS folder contains Python scripts for preprocessing, exploratory data analysis (EDA), and final clustering and modeling. The OUTPUTS folder contains figures generated from the analyses, described in detail below.  

---

## Software and Platforms

- **Primary Software:** Python 3.1  
- **Python Packages Used for Cleaning and Analysis:**  
  `pandas`, `matplotlib`, `seaborn`, `os`, `plotly.express`, `numpy`, `PIL`, `scikit-learn`, `pytesseract`, `pillow`  
- **Development Environment:**  
  - Linux  

---

## Documentation Map

This repository contains the following files and directories:

### **SCRIPTS** (Python scripts for preprocessing, EDA, and modeling)

- `analysis.py` – Extracts visual features from advertisement images using MobileNet, performs OCR to extract embedded text, engineers metadata features (contrast, color, text density), applies PCA and t-SNE for dimensionality reduction, clusters ads using KMeans, and applies LDA to text data. Outputs labeled datasets and visualizations.
- `preprocessing.py` – Loads and summarizes the structure of the Parquet dataset, providing column data types, non-null counts, memory usage, and basic numeric statistics. Helps in understanding the dataset’s initial structure.
- `eda.py` – Cleans ad metadata by parsing dimensions, removing invalid entries, calculating area and aspect ratios, and producing frequency visualizations for widths, heights, dimensions, and aspect ratios.
- `requirements.txt` – Lists all required Python packages for reproducibility.

### **DATA** (Parquet dataset — download separately)

- The dataset is too large to push via GitHub. It can be downloaded [here](https://drive.google.com/drive/folders/1NHBysZ5VgMlqOjqfnfftpxTf-JamfPjm?usp=drive_link).
- Each row represents a single online advertisement, including:
  - **Image:** Byte-encoded image
  - **Text:** Extracted text content
  - **Dimensions:** Tuple of width and height
 - 'Data Appendix - Project 3.pdf'

### **OUTPUTS** (Figures and cleaned outputs)

- `tsne.png` – A t-SNE visualization of ad embeddings colored by aspect ratios, highlighting clusters based on ad shape.
- `tsne1.png` – A second t-SNE visualization confirming clustering by dimensional properties, suggesting standard layout formats.
- `Top_aspect_ratios.png`, `top_heights.png`, `top_widths.png`, `top_dimensions.png` – EDA graphs visualizing the most common advertisement widths, heights, dimension pairs, and aspect ratios across the dataset.

### **LICENSE** and **README**

- `License.md` – Contains the MIT license for this repository.

### **REFERENCES**
- `References - Project 3.pdf` – Cited materials and datasets.

---

## Instructions for Reproducing Results

To reproduce the results from this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python packages using the `requirements.txt` file.
3. Download the Parquet dataset from [this link](https://drive.google.com/drive/folders/1NHBysZ5VgMlqOjqfnfftpxTf-JamfPjm?usp=drive_link).
4. Run `preprocessing.py` to inspect and understand the dataset structure.
5. Run `eda.py` to clean the metadata and generate EDA visualizations.
6. Run `analysis.py` to perform feature extraction, clustering, topic modeling, and generate detailed visualizations.

---

## References

[1] "Online Advertising: How to Create Effective Online Ad Campaigns," Wordstream. [Online]. Available: [https://www.wordstream.com/online-advertising](https://www.wordstream.com/online-advertising).

[2] PeterBrendan, "AdImageNet," Hugging Face. [Online]. Available: [https://huggingface.co/datasets/PeterBrendan/AdImageNet](https://huggingface.co/datasets/PeterBrendan/AdImageNet).

[3] K. Hanenko, "Widely used AI/ML models in image recognition," Altamira.ai. [Online]. Available: [https://www.altamira.ai/blog/widely-used-ai-ml-models-in-image-recognition/](https://www.altamira.ai/blog/widely-used-ai-ml-models-in-image-recognition/).

[4] cpowell229, "DSProject3/SCRIPTS/preprocessing.py," GitHub. [Online]. Available: [https://github.com/cpowell229/DSProject3/blob/main/SCRIPTS/preprocessing.py](https://github.com/cpowell229/DSProject3/blob/main/SCRIPTS/preprocessing.py).

