# Interpretable Multimodal Zero-Shot ECG Diagnosis via Structured Clinical Knowledge Alignment
<div align="center">

<div>
    <a href='https://tang-jia-lu.github.io/' target='_blank'>Jialu Tang</a><sup>1</sup>&emsp;
    <a href='https://manhph2211.github.io/' target='_blank'>Pham Hung Manh</a><sup>2</sup>&emsp;
    <a href='https://www.researchgate.net/profile/Ignace-De-Lathauwer' target='_blank'>Ignace L.J. De Lathauwer</a><sup>2</sup>&emsp;
    <a href='https://www.researchgate.net/profile/Ignace-De-Lathauwer' target='_blank'>Henk S. Schipper</a><sup>4</sup>&emsp;
    <a href=https://www.tue.nl/en/research/researchers/yuan-lu' target='_blank'>Yuan Lu</a><sup>1</sup>&emsp;
    <a href='https://www.dongma.info/' target='_blank'>Dong Ma</a><sup>2</sup>&emsp;   
    <a href='https://aqibsaeed.github.io/' target='_blank'>Aaqib Saeed</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup><a href="https://www.tue.nl/en/our-university/departments/industrial-design/research/our-research-labs/decentralized-artificial-intelligence-research-lab" target="_blank" rel="noopener noreferrer">
                        Eindhoven University of Technology, Netherland
                      </a>&emsp;

<sup>2</sup>Singapore Management University, Singapore;
<sup>3</sup>Maxima Medical Center, Netherland;
<sup>4</sup>Erasmus Medical Center, Netherland;

</div>
</div>

## 📝 Abstract
Electrocardiogram (ECG) interpretation is essential for cardiovascular disease diagnosis, but current automated systems often struggle with transparency and generalization to unseen conditions. To address this, we introduce ZETA, a zero-shot multimodal framework designed for interpretable ECG diagnosis aligned with clinical workflows. ZETA uniquely compares ECG signals against structured positive and negative clinical observations, which are curated through an LLM-assisted, expert-validated process, thereby mimicking differential diagnosis. Our approach leverages a pre-trained multimodal model to align ECG and text embeddings without disease-specific fine-tuning. Empirical evaluations demonstrate ZETA's competitive zero-shot classification performance and, importantly, provide qualitative and quantitative evidence of enhanced interpretability, grounding predictions in specific, clinically relevant positive and negative diagnostic features. ZETA underscores the potential of aligning ECG analysis with structured clinical knowledge for building more transparent, generalizable, and trustworthy AI diagnostic systems. We will release the curated observation dataset and code to facilitate future research.

<div align="left">
<img src="img/ZETA.png" width="80%">
<h3>📚 Three Features </h3>
</div>

- **🩺 Clinical Grounding:** Structured positive/negative observations emulate differential diagnosis, enhancing medical reasoning.
- **🧠 Expert-in-the-Loop Curation:** LLM-guided + expert-validated annotation pipeline ensures trustworthy and clinically relevant supervision.
- **📊 Transparent Evaluation:** Interpretability metrics and qualitative diagnostics support actionable insights beyond accuracy.

## 🔧 Requirements

### Dataset 

Datasets we used:
- **PTB-XL**: We downloaded the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset which consisting four subsets, Superclass, Subclass, Form, Rhythm.

- **CPSC2018**: We downloaded the [CPSC2018](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/) dataset which consisting three training sets. 

- **CSN(Chapman-Shaoxing-Ningbo)**: We downloaded the [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/) dataset.

### Observations Dataset
ZETA operates by comparing ECG representations against structured, validated natural language descriptions of diagnostic observations. Our approach frames the zero-shot problem as a comparison between ECG signals and predefined sets of positive and negative textual observations associated with specific conditions.
The generated observations undergo review and validation by a doctor (specializing in cardiology). This human-in-the-loop step is critical for ensuring the clinical accuracy, relevance, and interpretability of the structured knowledge used by ZETA. The expert evaluates each candidate observation based on four key dimensions

<example here! full in config:observation>
## 💡 Running scripts

To prepare your experiment, please setup your configuration at the main.py. You can configure the specific federated learning strategy at server.py. You can simply execute the main script them to run the experiment, the results will save as a `logs` file.

```
cd ./ZETA
python main.py
```


## Citing ZETA

```bibtex

```
