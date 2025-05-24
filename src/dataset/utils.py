import logging
import re

import numpy as np
from collections import defaultdict

import pandas as pd

from src.dataset.movie_review_dataset import MergedReviewDataset

logging.basicConfig(level=logging.INFO)


def show_patters(merged_df):
    unique_scores = merged_df['review_score'].dropna().astype(str).str.strip().unique()
    # group scores by detected pattern
    pattern_groups = defaultdict(list)
    for score in unique_scores:
        pattern = detect_pattern(score)
        pattern_groups[pattern].append(score)

    # show results
    for pattern, scores in pattern_groups.items():
        print(f"\nPattern: {pattern} ({len(scores)} types)")
        for s in sorted(scores)[:10]:
            print(f"  {s}")
        if len(scores) > 10:
            print("  ...")


def detect_pattern(score):
    score = score.strip().upper()

    if re.fullmatch(r'\d+(\.\d+)?/\d+(\.\d+)?', score):       # e.g. 3.5/5, 7/10
        return 'fraction'
    elif re.fullmatch(r'\d{1,3}/100', score):                 # e.g. 85/100
        return 'percentage'
    elif re.fullmatch(r'\d+(\.\d+)?', score):                 # e.g. 7, 9.5
        return 'numeric'
    elif re.fullmatch(r'[A-F][+-]?', score):                  # e.g. A, B+, C-
        return 'letter_grade'
    elif re.fullmatch(r'.*OUT OF \d+', score):                # e.g. 3 out of 5
        return 'textual_fraction'
    elif re.fullmatch(r'[A-Z\s]+STARS?', score):              # e.g. THREE STARS
        return 'star_rating'
    elif re.fullmatch(r'[A-Z][A-Z\s]*', score):               # e.g. RECOMMENDED, AVOID
        return 'descriptive_text'
    else:
        return 'unknown'


def normalise_score(score):
    try:
        # handle fractional formats like "3.5/4" or "8/10"
        if '/' in score:
            parts = re.split(r'[^\d.]+', score)
            if len(parts) == 2 and parts[0] and parts[1]:
                numerator, denominator = float(parts[0]), float(parts[1])
                return round((numerator / denominator) * 10, 2)  # Scale to 0-10

        # handle percentages like "85/100"
        if re.match(r'\d{1,3}/100', score):
            percentage = float(score.split('/')[0])
            return round((percentage / 10), 2)  # convert percentage to 0-10 scale

        # handle numeric scores like "7.5", "8", "5.6"
        if re.match(r'^\d+(\.\d+)?$', score):
            value = float(score)
            # scale values greater than 10 (e.g., "85/100" already handled) to 0-10
            return value if value <= 10 else round(value / 10, 2)

        # handle letter grades
        letter_grades = {
            'A+': 10, 'A': 9.5, 'A-': 9,
            'B+': 8.5, 'B': 8, 'B-': 7.5,
            'C+': 7, 'C': 6.5, 'C-': 6,
            'D+': 5.5, 'D': 5, 'D-': 4.5,
            'F': 2, 'F-': 1
        }
        if score.strip().upper() in letter_grades:
            return letter_grades[score.strip().upper()]

        # handle descriptive phrases
        descriptive_scores = {
            'FIVE STARS': 10, 'FOUR STARS': 8, 'THREE STARS': 6,
            'TWO STARS': 4, 'ONE STAR': 2, 'ZERO STARS': 0,
            'Highly Recommended': 9, 'Not Recommended': 2,
            'Recommended': 7, 'Avoid': 1, 'Catch It On Cable': 5
        }
        if score.strip().lower() in [key.lower() for key in descriptive_scores]:
            return descriptive_scores[score.strip()]

        # handle scores with mixed text, e.g., "3 out of 5"
        if "out of" in score:
            parts = re.findall(r'\d+', score)
            if len(parts) == 2:
                numerator, denominator = map(float, parts)
                return round((numerator / denominator) * 10, 2)
    except:
        pass

    # return NaN for unprocessable scores
    return np.nan


def refine_normalised_score(score: str) -> float:
    try:
        score = str(score).strip()
        if '/' in score:
            parts = re.findall(r'\d+(\.\d+)?', score)
            if len(parts) == 2:
                numerator, denominator = map(float, parts)
                if denominator in [1, 2, 4, 5, 10, 100]:
                    return round((numerator / denominator) * 10, 2)
        return np.nan
    except Exception as e:
        print(f"Error re-normalising score '{score}': {e}")
        return np.nan


def normalise_scores(merged_review_ds: MergedReviewDataset) -> MergedReviewDataset:
    if merged_review_ds.df is None:
        raise ValueError("Call merge() before normalise_scores()")

    df = merged_review_ds.df.copy()

    df = df[['review_score', 'movie_title', 'movie_info', 'review_content']].copy()
    df['normalised_review_score'] = df['review_score'].astype(str).apply(normalise_score)
    df = df.dropna(subset=['normalised_review_score'])
    df['normalised_review_score'] = df['normalised_review_score'].astype(float)

    # clean-up: filter invalid scores
    df = df[
        (df['normalised_review_score'] >= 0) &
        (df['normalised_review_score'] <= 10) &
        (~df['normalised_review_score'].isna())
    ].reset_index(drop=True)

    merged_review_ds.df = df
    logging.info(f"Normalised and filtered scores for {len(df)} rows")
    return merged_review_ds


def balance_classes(merged_review_ds: MergedReviewDataset) -> MergedReviewDataset:
    """
    Normalise `normalised_review_score` into integer buckets 1–10 using min-max scaling,
    filter out short reviews, and balance all classes by undersampling
    to the smallest class size.
    """
    if merged_review_ds.df is None or 'normalised_review_score' not in merged_review_ds.df.columns:
        raise ValueError("Call normalise_scores() before balance_classes()")

    df = merged_review_ds.df.copy()

    # 1. min-max scale final_score → 1–10 integer buckets
    min_score = df['normalised_review_score'].min()
    max_score = df['normalised_review_score'].max()

    df['normalised_review_score'] = (1 + 9 * (df['normalised_review_score'] - min_score) / (max_score - min_score)).round().astype(int)

    # 2. filter out short reviews (<64 characters)
    df = df[df['review_content'].str.len() >= 64]

    # 3. get the smallest class size
    min_count = df['normalised_review_score'].value_counts().min()

    # 4. undersample all score classes to match min_count
    df_balanced = (
        df.groupby('normalised_review_score')
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    logging.info(f"Balanced dataset to {len(df_balanced)} rows ({min_count} per score class)")
    merged_review_ds.df = df_balanced
    return merged_review_ds

