import pandas as pd
import logging
from typing import Optional
import re
import redshift_connector
import boto3, io

logging.basicConfig(level=logging.INFO)


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).strip())


def clean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df = df[df[col].notnull()]
        df[col] = df[col].apply(clean_text)
    return df


class BaseDataset:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this.")

    def __len__(self):
        return len(self.df) if self.df is not None else 0


class LocalCriticReviewDataset(BaseDataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        df = clean_columns(df, ['rotten_tomatoes_link', 'review_score', 'review_content'])
        self.df = df
        logging.info(f"Loaded {len(df)} critic reviews from local CSV")
        return self.df


class LocalMovieMetadataDataset(BaseDataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        df = clean_columns(df, ['rotten_tomatoes_link', 'movie_title', 'movie_info'])
        self.df = df
        logging.info(f"Loaded {len(df)} movie metadata rows from local CSV")
        return self.df


class S3CriticReviewDataset(BaseDataset):
    def __init__(self, s3_uri: str):
        super().__init__()
        self.bucket, self.key = self._parse_s3_uri(s3_uri)

    def _parse_s3_uri(self, uri: str):
        parts = uri.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    def load(self):
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=self.bucket, Key=self.key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        df = clean_columns(df, ['rotten_tomatoes_link', 'review_score', 'review_content'])
        self.df = df
        logging.info(f"Loaded {len(df)} critic reviews from S3")
        return self.df


class S3MovieMetadataDataset(BaseDataset):
    def __init__(self, s3_uri: str):
        super().__init__()
        self.bucket, self.key = self._parse_s3_uri(s3_uri)

    def _parse_s3_uri(self, uri: str):
        parts = uri.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    def load(self):
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=self.bucket, Key=self.key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        df = clean_columns(df, ['rotten_tomatoes_link', 'movie_title', 'movie_info'])
        self.df = df
        logging.info(f"Loaded {len(df)} movie metadata rows from S3")
        return self.df


class RedshiftCriticReviewDataset(BaseDataset):
    def __init__(self, table: str, credentials: dict):
        super().__init__()
        self.table = table
        self.credentials = credentials

    def load(self):
        conn = redshift_connector.connect(
            host=self.credentials['host'],
            database=self.credentials['database'],
            port=self.credentials.get('port', 5439),
            user=self.credentials['user'],
            password=self.credentials['password']
        )
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {self.table}")
        df = cursor.fetch_dataframe()

        cursor.close()
        conn.close()

        df = clean_columns(df, ['rotten_tomatoes_link', 'review_score', 'review_content'])
        self.df = df

        logging.info(f"Loaded {len(df)} critic reviews from Redshift")
        return self.df


class RedshiftMovieMetadataDataset(BaseDataset):
    def __init__(self, table: str, credentials: dict):
        super().__init__()
        self.table = table
        self.credentials = credentials

    def load(self):
        conn = redshift_connector.connect(
            host=self.credentials['host'],
            database=self.credentials['database'],
            port=self.credentials.get('port', 5439),
            user=self.credentials['user'],
            password=self.credentials['password']
        )
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {self.table}")
        df = cursor.fetch_dataframe()

        cursor.close()
        conn.close()

        df = clean_columns(df, ['rotten_tomatoes_link', 'movie_title', 'movie_info'])
        self.df = df

        logging.info(f"Loaded {len(df)} movie metadata rows from Redshift")
        return self.df


class MergedReviewDataset:
    def __init__(self, critics_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.critics_df = critics_df
        self.movies_df = movies_df
        self.df: Optional[pd.DataFrame] = None

    def merge(self) -> pd.DataFrame:
        df = self.critics_df.merge(self.movies_df, on='rotten_tomatoes_link', how='left')
        df = df[df['review_score'].notnull() & df['review_score'].str.strip().astype(bool)]
        self.df = df
        logging.info(f"Merged dataset contains {len(df)} rows")
        return self.df

    def preview(self, columns=None, n=5):
        if columns is None:
            columns = ['review_score', 'movie_title', 'movie_info', 'review_content']
        if self.df is None:
            raise ValueError("Must call merge() before preview().")
        return self.df[columns].head(n)

    def __len__(self):
        return len(self.df) if self.df is not None else 0
