# Cleaning_Identification

## Overview

This directory contains a speech-cleaning and identification pipeline. The preprocessing folder contains the full speech system; holds the cleaning/identification artifacts and runs the pipeline and trains/loads the identification model.

## Contents

- [Cleaning_Identification/label_speakers.py](Cleaning_Identification/label_speakers.py) - Cleaning script with premade model implementation by MComp student ...
- [Cleaning_Identification/CLEAN_agent_donor.joblib](Cleaning_Identification/CLEAN_agent_donor.joblib) — saved joblib pipeline/model (sklearn Pipeline + TfidfVectorizer, etc.).

## Quick start

1. Run label_speakers.py (which produces data-human.csv), before running fuzzy_embeddings.py, to get the final output
2. Outputs are `sentence_embeddings.jsonl` (primary file that is used for the next step of model pipeline) and `conversation_turns.jsonl` (secondary file for us to check on data in natural language).
