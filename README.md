---
title: Systematic Error Analysis and Labeling
emoji: 🦭
colorFrom: yellow
colorTo: pink
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
pinned: false
license: apache-2.0
---
# SEAL
Systematic Error Analysis and Labeling (SEAL) is an interactive tool for discovering systematic errors in NLP models via clustering on high-loss example groups and semantic labeling for interpretability of those error-groups. It supports fine-grained analytical visualization for interactively zooming into potential systematic bugs and features for crafting prompts to label those bugs semantically.

🎥 [Demo screencast](https://vimeo.com/736659216)

<p>
    <img src="./assets/website/seal.gif" alt="Demo gif"/>
</p>

## Table of Contents
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running Locally](#running-locally)
- [Citation](#citation)

## Installation
Please use python>=3.8 since some dependencies require that for installation.
```shell
git clone https://huggingface.co/spaces/nazneen/seal
cd seal
pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart
```
streamlit run app.py
```

## Running Locally

## Citation
