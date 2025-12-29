# Cleaning_Identification

## Overview

This directory contains a speech-cleaning and identification pipeline. The preprocessing folder contains the full speech system; holds the cleaning/identification artifacts and runs the pipeline and trains/loads the identification model.

## Contents

- [Cleaning_Identification/label_speakers.py](Cleaning_Identification/label_speakers.py) - Cleaning script with premade model implementation by MComp student ...
- [Cleaning_Identification/CLEAN_agent_donor.joblib](Cleaning_Identification/CLEAN_agent_donor.joblib) — saved joblib pipeline/model (sklearn Pipeline + TfidfVectorizer, etc.).
- [Cleaning_Identification/dummydataset.csv](Cleaning_Identification/dummydataset.csv) — example CSV used by the notebook.

## Quick start

1. Run label_speakers.py before running fuzzy_embeddings.py
