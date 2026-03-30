import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from rapidfuzz import process, fuzz
from collections import defaultdict
import re
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif


def load_all_csv_files(data_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a folder into a dictionary of DataFrames.

    Args:
        data_folder: Path to the folder containing CSV files

    Returns:
        Dictionary with filename as key and DataFrame as value
    """
    data_path = Path(data_folder) if isinstance(data_folder, str) else data_folder
    dataframes = {}

    if not data_path.exists():
        raise FileNotFoundError(f"Folder {data_folder} does not exist")

    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return dataframes

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dataframes[file_path.stem] = df
            print(f"Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    return dataframes


def concatenate_all_files(
    data_folder: str,
    file_pattern: str = "*.csv",
    column_mappings: Dict[str, Dict[str, str]] = None,
    file_order: list = None
) -> pd.DataFrame:
    """
    Load and concatenate all files with custom column name mapping.

    The first file serves as the base schema. Subsequent files are mapped
    to match the base schema using the provided column mappings.

    Args:
        data_folder: Path to the folder containing files
        file_pattern: Glob pattern for files to load (default: "*.csv")
        column_mappings: Dictionary mapping file names (without extension) to column mappings.
            Format: {
                'file1': {'original_title': 'title', 'movie_genre': 'genre'},
                'file2': {'film_title': 'title', 'category': 'genre'}
            }
            The first file is not mapped and serves as the base schema.
        file_order: Optional list of filenames (without extension) to specify order.
            If None, files are processed alphabetically.

    Returns:
        Single concatenated DataFrame
    """
    data_path = Path(data_folder) if isinstance(data_folder, str) else data_folder

    if not data_path.exists():
        raise FileNotFoundError(f"Folder {data_folder} does not exist")

    files = list(data_path.glob(file_pattern))

    if not files:
        raise ValueError(f"No files matching {file_pattern} found in {data_folder}")

    files_dict = {file_path.stem: file_path for file_path in files}

    if file_order:
        file_names = []
        for name in file_order:
            if name in files_dict:
                file_names.append(name)
            else:
                print(f"Warning: File '{name}' not found in {data_folder}")
        remaining = [name for name in sorted(files_dict.keys()) if name not in file_names]
        file_names.extend(remaining)
    else:
        file_names = sorted(files_dict.keys())

    if not file_names:
        raise ValueError("No valid files to process")

    print("=" * 80)
    print(f"Processing {len(file_names)} files in order:")
    for i, name in enumerate(file_names, 1):
        print(f"  {i}. {name}")
    print("=" * 80)

    base_name = file_names[0]
    base_file = files_dict[base_name]

    print(f"\n[BASE] Loading {base_file.name}...")

    if base_file.suffix == '.csv':
        base_df = pd.read_csv(base_file)
    elif base_file.suffix == '.parquet':
        base_df = pd.read_parquet(base_file)
    elif base_file.suffix == '.json':
        base_df = pd.read_json(base_file)

    print(f"  Shape: {base_df.shape}")
    print(f"  Columns: {base_df.columns.tolist()}")

    result_df = base_df.copy()
    base_columns = set(base_df.columns)

    for file_name in file_names[1:]:
        file_path = files_dict[file_name]

        print(f"\n[{file_name}] Loading {file_path.name}...")

        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                continue

            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            if column_mappings and file_name in column_mappings:
                mapping = column_mappings[file_name]
                print(f"  Applying column mappings: {mapping}")
                df = df.rename(columns=mapping)

            new_columns = set(df.columns) - base_columns
            if new_columns:
                print(f"  New columns found: {list(new_columns)}")
                for col in new_columns:
                    result_df[col] = pd.NA
                base_columns.update(new_columns)

            missing_columns = base_columns - set(df.columns)
            if missing_columns:
                print(f"  Missing columns (will be filled with NaN): {list(missing_columns)}")
                for col in missing_columns:
                    df[col] = pd.NA

            df = df[result_df.columns]
            result_df = pd.concat([result_df, df], ignore_index=True)
            print(f"  After concatenation: {result_df.shape}")

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    print("\n" + "=" * 80)
    print(f"Final DataFrame: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
    print(f"Columns: {result_df.columns.tolist()}")
    print("=" * 80)

    return result_df


class DataCleaner:
    LIST_LIKE_COLUMNS = [
        # 'genres',  ## target — kept out intentionally
        'production_companies',
        'production_countries',
        'spoken_languages',
        'keywords',
        'directors',
        'writers',
        'cast'
    ]

    BOOLEAN_COLS = ['adult']
    NUMERIC_COLS = [
        'vote_average',
        'vote_count',
        'revenue',
        'runtime',
        'budget',
        'popularity',
        'averageRating',
        'numVotes',
        'num_reviews',
        'year'
    ]

    # Columns with no predictive value for genre — dropped early
    COLUMNS_TO_DROP = [
        'id', 'backdrop_path', 'poster_path', 'homepage',
        'tconst', 'original_title'
    ]

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        self._original_df = df.copy()
        self.current_df = df.copy()
        self.verbose = verbose

    def _vprint(self, *args, **kwargs):
        """Internal helper to print only if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    def audit_column_health(self):
        """
        Returns a detailed summary of column health, catching all null variants.
        """
        audit_data = []
        for col in self.current_df.columns:
            series = self.current_df[col]
            dtype = str(series.dtype)

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                clean_s = series.astype(str).str.strip().str.lower()
                nan_count = (series.isna() | (clean_s == 'nan') | (clean_s == '')).sum()
            else:
                nan_count = series.isna().sum()

            audit_data.append({
                'column': col,
                'type': dtype,
                'missing_or_nan': nan_count,
                'total_rows': len(series)
            })
        return pd.DataFrame(audit_data)

    def drop_empty_genres(self):
        """Drops rows where the 'genres' column is null, empty, or 'nan'."""
        if 'genres' not in self.current_df.columns:
            self._vprint("⚠️ Warning: 'genres' column not found. Skipping drop.")
            return self.current_df

        clean_s = self.current_df['genres'].astype(str).str.strip().str.lower()
        empty_mask = (self.current_df['genres'].isna() | (clean_s == 'nan') | (clean_s == ''))

        initial_count = len(self.current_df)
        self.current_df = self.current_df[~empty_mask].copy()

        dropped_count = initial_count - len(self.current_df)
        self._vprint(f"✓ Dropped {dropped_count} rows with empty 'genres'")

        return self.current_df

    def drop_useless_columns(self):
        """
        Drops columns with no predictive value for genre prediction.
        These are identifiers, image paths, and redundant fields.
        """
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP if c in self.current_df.columns]
        self.current_df.drop(columns=cols_to_drop, inplace=True)
        self._vprint(f"✓ Dropped {len(cols_to_drop)} useless columns: {cols_to_drop}")
        return self.current_df

    def standardise_column_names(self):
        """
        Standardises column names by removing control characters and extra spaces.
        Logs every specific change made to the headers for audit purposes.
        """
        self._vprint("\n" + "=" * 80)
        self._vprint("-> STARTING COLUMN NAME STANDARDISATION")
        self._vprint("=" * 80)

        old_cols = self.current_df.columns.tolist()
        new_cols = []
        changes = []

        for col in old_cols:
            cleaned = "".join(char if char.isprintable() else ' ' for char in col)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            new_cols.append(cleaned)

            if cleaned != col:
                changes.append({
                    'original': repr(col),
                    'standardised': cleaned
                })

        self.current_df.columns = new_cols

        if not changes:
            self._vprint("✓ All column names were already standard. No changes made.")
        else:
            self._vprint(f"⚠️ Found and corrected {len(changes)} non-standard column names:")
            self._vprint(f"\n{'ORIGINAL (RAW)':<50} | {'STANDARDISED':<30}")
            self._vprint("-" * 80)
            for change in changes:
                self._vprint(f"{change['original']:<50} | {change['standardised']:<30}")

        self._vprint("=" * 80 + "\n")
        return self.current_df

    def standardise_data(self):
        """
        Applies three-step cleaning: lowercase, strip whitespace, and unify NaNs.
        """
        string_cols = self.current_df.select_dtypes(include=['object', 'string']).columns

        for col in string_cols:
            self.current_df[col] = (self.current_df[col]
                                    .astype(str)
                                    .str.strip()
                                    .str.lower()
                                    .replace(['nan', 'none', ''], np.nan))

        self._vprint(f"✓ Standardised {len(string_cols)} columns (lowercase, stripped, null-unified)")
        return self.current_df

    def profile_unique_values(self):
        """Profiles columns to find categories, numeric ranges, or median list item counts."""
        profile_results = []

        for col in self.current_df.columns:
            all_unique = self.current_df[col].unique()
            series_no_na = self.current_df[col].dropna()
            clean_unique = series_no_na.unique()

            if len(all_unique) < 2:
                continue

            row_data = {
                'column': col,
                'type': str(self.current_df[col].dtype),
                'unique_count': len(clean_unique)
            }

            if pd.api.types.is_numeric_dtype(self.current_df[col]):
                row_data['analysis_type'] = 'Numeric'
                row_data['detail'] = f"Min: {series_no_na.min()} | Max: {series_no_na.max()}"
                row_data['example'] = str(series_no_na.iloc[0]) if not series_no_na.empty else "NaN"

            elif col in self.LIST_LIKE_COLUMNS:
                row_data['analysis_type'] = 'List-like'
                item_counts = series_no_na.apply(
                    lambda x: len(str(x).split(',')) if str(x).lower() != 'unset' else 0
                )

                if not item_counts.empty:
                    median_len = int(item_counts.median())
                    row_data['detail'] = f"Median elements: {median_len} (Max: {int(item_counts.max())})"
                    median_match = item_counts[item_counts == median_len].index
                    row_data['example'] = f"Typical: {series_no_na.loc[median_match[0]]}"
                else:
                    row_data['detail'] = "No valid list items"
                    row_data['example'] = "N/A"

            else:
                row_data['analysis_type'] = 'Categorical'
                row_data['detail'] = "Textual categories"
                row_data['example'] = str(clean_unique[0]) if len(clean_unique) > 0 else "NaN"

            profile_results.append(row_data)

        return pd.DataFrame(profile_results)

    def find_fuzzy_duplicates(self, threshold: int = 85, min_length: int = 4):
        """
        Finds groups of similar strings in textual columns.
        Ignores columns explicitly defined as list-like to avoid noise.
        """
        issue_list = []
        potential_cols = [c for c in self.current_df.select_dtypes(include=['object', 'string']).columns
                          if c not in self.LIST_LIKE_COLUMNS]

        for col in potential_cols:
            unique_values = [
                str(x) for x in self.current_df[col].dropna().unique()
                if len(str(x)) >= min_length
            ]

            if len(unique_values) < 2 or len(unique_values) > 1000:
                continue

            duplicate_groups = []
            processed_values = set()

            for i, val in enumerate(unique_values):
                if val in processed_values:
                    continue

                matches = process.extract(
                    val,
                    unique_values[i+1:],
                    scorer=fuzz.ratio,
                    score_cutoff=threshold
                )

                if matches:
                    group = [val] + [match[0] for match in matches]
                    duplicate_groups.append(sorted(group))
                    processed_values.update(group)

            if duplicate_groups:
                issue_list.append({
                    'column': col,
                    'found_groups': len(duplicate_groups),
                    'examples': duplicate_groups[:3]
                })

        return pd.DataFrame(issue_list)

    def drop_highly_empty_columns(self, threshold=0.8, empty_values=[]):
        """
        Drops columns where >threshold % of values are empty/null/unset.

        Args:
            threshold: Fraction of empty values above which a column is dropped (default 0.8)
            empty_values: Additional string values to treat as empty

        Returns:
            list: Names of dropped columns
        """
        dropped_cols = []

        # BUG FIX: was self.df (does not exist) — corrected to self.current_df
        for col in self.current_df.columns:
            total_rows = len(self.current_df)
            empty_mask = (
                self.current_df[col].isna() |
                self.current_df[col].astype(str).str.lower().isin([v.lower() for v in empty_values])
            )
            empty_count = empty_mask.sum()
            empty_pct = empty_count / total_rows

            if empty_pct > threshold:
                self.current_df = self.current_df.drop(columns=[col])
                dropped_cols.append(col)
                self._vprint(f"-> Dropped '{col}': {empty_pct:.1%} empty ({empty_count}/{total_rows})")

        self._vprint(f"\nDropped {len(dropped_cols)} null columns in total.")
        return dropped_cols

    def standardise_numeric(self, columns: list = None):
        """
        Converts specified columns to numeric types.
        Handles 'dirty' strings by converting them to NaN.
        """
        cols_to_fix = columns or self.NUMERIC_COLS

        for col in [c for c in cols_to_fix if c in self.current_df.columns]:
            self._vprint(f"-> Converting to numeric: '{col}'")
            self.current_df[col] = pd.to_numeric(self.current_df[col], errors='coerce')
            self.current_df.loc[self.current_df[col] == -999, col] = np.nan

        self._vprint(f"✓ Standardised {len(cols_to_fix)} numeric columns")
        return self.current_df

    def standardise_boolean(self, columns: list = None):
        """
        Converts specified columns to boolean (True/False).
        Converts all null-like values (NaN, None, 'nan') to False.
        """
        cols_to_fix = columns or self.BOOLEAN_COLS

        bool_map = {
            'true': True, '1': True, 1: True, 1.0: True, 't': True, 'yes': True,
            'false': False, '0': False, 0: False, 0.0: False, 'f': False, 'no': False
        }

        for col in [c for c in cols_to_fix if c in self.current_df.columns]:
            self._vprint(f"-> Converting to boolean: '{col}'")
            self.current_df[col] = (self.current_df[col]
                                    .astype(str)
                                    .str.strip()
                                    .str.lower()
                                    .map(bool_map)
                                    .fillna(False)
                                    .astype(bool))

        self._vprint(f"✓ Standardised {len(cols_to_fix)} boolean columns")
        return self.current_df

    def convert_strings_to_lists(self):
        """
        Normalises list-like columns (e.g., 'cast', 'directors') without sorting.
        Maintains original order (e.g., lead actor stays first).
        """
        delimiter_pattern = r'[\+/\|;]'

        for col in [c for c in self.LIST_LIKE_COLUMNS if c in self.current_df.columns]:
            self._vprint(f"-> Normalising order-sensitive list: '{col}'")

            def process_ordered_string(s):
                if not isinstance(s, str) or not s.strip() or s.strip().lower() == 'nan':
                    return 'unset'
                items = [item.strip() for item in re.split(delimiter_pattern, s) if item.strip()]
                unique_ordered_items = list(dict.fromkeys(items))
                result = ", ".join(unique_ordered_items)
                return result if result else 'unset'

            self.current_df[col] = self.current_df[col].apply(process_ordered_string)

        self._vprint(f"✓ Normalised {len(self.LIST_LIKE_COLUMNS)} list columns (order preserved)")
        return self.current_df

    def expand_list_columns(self, columns: list = None):
        """
        Splits list-like columns into 3 separate columns (e.g., cast_1, cast_2, cast_3).
        Ignores 'genres' as it is the target.
        """
        # BUG FIX: was c != 'genre' — corrected to c != 'genres' to match actual column name
        cols_to_expand = columns or [c for c in self.LIST_LIKE_COLUMNS if c != 'genres']

        for col in [c for c in cols_to_expand if c in self.current_df.columns]:
            self._vprint(f"-> Expanding list-like column: '{col}'")

            expanded = self.current_df[col].apply(
                lambda x: str(x).split(', ') if str(x).lower() != 'unset' else []
            )

            for i in range(1, 4):
                new_col_name = f"{col}_{i}"
                self.current_df[new_col_name] = expanded.apply(
                    lambda x: x[i-1].strip() if len(x) >= i else np.nan
                )

            self.current_df.drop(columns=[col], inplace=True)

        self._vprint(f"✓ Expanded {len(cols_to_expand)} columns into triplet features")
        return self.current_df
    
    def analyse_mutual_information(self, top_n: int = 20, training=False):
        """Calculates and plots mutual information scores."""
        # This function is adapted from your original notebook [1]
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        X_prepared = X.copy()
        discrete_mask = []
        for col in X_prepared.columns:
            if X_prepared[col].dtype == 'object':
                X_prepared[col] = X_prepared[col].astype('category').cat.codes
                discrete_mask.append(True)
            else: # Assumes numeric columns are continuous
                X_prepared[col] = X_prepared[col].fillna(-999)
                discrete_mask.append(False)
        
        y_encoded = LabelEncoder().fit_transform(y)
        
        mi_scores = mutual_info_classif(X_prepared, y_encoded, discrete_features=discrete_mask, random_state=42)
        mi_df = pd.DataFrame({'feature': X.columns, 'mutual_information': mi_scores})
        mi_df = mi_df.sort_values('mutual_information', ascending=False)
        
        if not training:
            self._vprint(f"\n--- Top {top_n} Features (Mutual Information) ---")
        top_features = mi_df.head(top_n)['feature'].tolist()
        if not training:
            self._vprint(mi_df.head(top_n).to_string(index=False))
        
            fig = px.bar(mi_df.head(top_n), x='mutual_information', y='feature', orientation='h',
                         title=f'Top {top_n} Features by Mutual Information',
                         labels={'mutual_information': 'Mutual Information Score', 'feature': 'Feature'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            fig.show()
        
        return top_features


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Transforms the cleaned DataFrame into a feature matrix ready for BernoulliNB.

    Strategy:
    - Numeric columns    → binned into quantiles, then one-hot encoded (BernoulliNB needs binary input)
    - Low-cardinality    → one-hot encoded directly (status, original_language, movie_rated, adult)
    - List-like columns  → frequency-based binary flags for the top-N most common values
    - Text columns       → hand-crafted binary/count features (no embeddings, no TF-IDF)
    - genres             → split into genre_1 / genre_2 / genre_3 (top 3 in original order),
                           then label-encoded. genre_2 and genre_3 use 'none' for missing values.
    """

    # Genre-signaling keyword lists for hand-crafted text features
    GENRE_KEYWORDS = {
        'action':   ['explosion', 'fight', 'battle', 'war', 'chase', 'combat', 'attack', 'shoot'],
        'comedy':   ['funny', 'laugh', 'humor', 'hilarious', 'joke', 'comic', 'absurd'],
        'romance':  ['love', 'heart', 'relationship', 'marriage', 'affair', 'romantic', 'kiss'],
        'horror':   ['fear', 'terror', 'ghost', 'monster', 'kill', 'death', 'blood', 'dark'],
        'thriller': ['mystery', 'secret', 'suspense', 'tension', 'danger', 'escape', 'trap'],
        'sci_fi':   ['space', 'alien', 'future', 'robot', 'planet', 'technology', 'universe'],
        'drama':    ['family', 'struggle', 'life', 'journey', 'story', 'emotional', 'loss'],
        'crime':    ['murder', 'detective', 'police', 'crime', 'criminal', 'investigation', 'gang'],
    }

    # Placeholder for missing genre slots (genre_2, genre_3 when a movie has fewer genres)
    NONE_LABEL = 'none'

    # Columns to one-hot encode directly (low cardinality categoricals)
    LOW_CARDINALITY_COLS = ['status', 'original_language', 'movie_rated']

    # Numeric columns to bin then encode
    NUMERIC_TO_BIN = [
        'runtime', 'budget', 'revenue', 'popularity',
        'vote_average', 'vote_count', 'averageRating',
        'numVotes', 'num_reviews', 'year'
    ]

    # List-like columns for frequency-based binary flags
    LIST_LIKE_COLS = [
        'production_companies', 'production_countries',
        'spoken_languages', 'keywords',
        'directors', 'writers', 'cast'
    ]

    def __init__(self, top_n_per_list_col: int = 50, n_bins: int = 5, verbose: bool = True):
        """
        Args:
            top_n_per_list_col: How many top values to keep per list-like column
            n_bins: Number of quantile bins for numeric columns
            verbose: Print progress
        """
        self.top_n_per_list_col = top_n_per_list_col
        self.n_bins = n_bins
        self.verbose = verbose

        # These are fit on training data and reused on test data
        self._top_values_per_col: Dict[str, List[str]] = {}
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._ohe_categories: Dict[str, List[str]] = {}
        self._feature_columns: Optional[List[str]] = None

        # One LabelEncoder per genre slot — fitted on training labels only
        self._label_encoders: Dict[str, LabelEncoder] = {
            'genre_1': LabelEncoder(),
            'genre_2': LabelEncoder(),
            'genre_3': LabelEncoder(),
        }

        self._fitted = False

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    # ------------------------------------------------------------------
    # TARGET: genres → genre_1 / genre_2 / genre_3
    # ------------------------------------------------------------------

    def _split_genres(self, genres_series: pd.Series) -> pd.DataFrame:
        """
        Takes the raw genres string column and produces three columns:
          genre_1 — primary genre (always filled)
          genre_2 — secondary genre, or NONE_LABEL if absent
          genre_3 — tertiary genre, or NONE_LABEL if absent

        Strategy: take the first 3 genres in the original order (preserving
        the primary genre signal), then fill missing slots with NONE_LABEL.
        We deliberately do NOT sort here because order is meaningful
        (the first genre is the principal genre for that movie).
        """
        def parse_top3(val):
            if pd.isna(val) or str(val).strip().lower() in ('nan', '', 'unset'):
                return [self.NONE_LABEL, self.NONE_LABEL, self.NONE_LABEL]
            genres = [g.strip() for g in str(val).split(',') if g.strip()]
            # Keep only the first 3 — ignore anything beyond position 3
            top3 = genres[:3]
            # Pad with NONE_LABEL to always have exactly 3 slots
            while len(top3) < 3:
                top3.append(self.NONE_LABEL)
            return top3

        parsed = genres_series.apply(parse_top3)
        return pd.DataFrame(
            parsed.tolist(),
            columns=['genre_1', 'genre_2', 'genre_3'],
            index=genres_series.index
        )

    def fit_target(self, genres_series: pd.Series):
        """
        Fits one LabelEncoder per genre slot on training data.
        NONE_LABEL is explicitly added to every slot's vocabulary so that
        transform() never crashes when genre_2 or genre_3 is absent in a row,
        even if the training split happened to contain no such rows.
        """
        split = self._split_genres(genres_series)
        for slot, le in self._label_encoders.items():
            # Concatenate a sentinel Series containing NONE_LABEL before fitting,
            # guaranteeing it is always a known class regardless of the split.
            values = pd.concat([
                split[slot],
                pd.Series([self.NONE_LABEL])
            ], ignore_index=True)
            le.fit(values)
            self._vprint(f"  '{slot}': {len(le.classes_)} classes — {list(le.classes_)}")

    def transform_target(self, genres_series: pd.Series) -> pd.DataFrame:
        """
        Transforms genres into a 3-column integer DataFrame (genre_1, genre_2, genre_3).
        Unseen labels at test time are mapped to NONE_LABEL (safe fallback).
        """
        split = self._split_genres(genres_series)
        result = pd.DataFrame(index=genres_series.index)

        for slot, le in self._label_encoders.items():
            known_classes = set(le.classes_)
            # Map unseen labels to NONE_LABEL so transform never crashes
            safe = split[slot].apply(
                lambda x: x if x in known_classes else self.NONE_LABEL
            )
            result[slot] = le.transform(safe)

        return result

    def get_genre_classes(self) -> Dict[str, np.ndarray]:
        """Returns the class arrays for each genre slot (useful for reporting)."""
        return {slot: le.classes_ for slot, le in self._label_encoders.items()}

    # ------------------------------------------------------------------
    # FEATURES: numeric → binned → binary
    # ------------------------------------------------------------------

    def _bin_numeric_column_fit(self, series: pd.Series, col: str):
        """Fits bin edges on a numeric series using quantiles."""
        clean = series.dropna()
        if clean.empty or clean.nunique() < self.n_bins:
            self._bin_edges[col] = None
            return
        _, edges = pd.qcut(clean, q=self.n_bins, retbins=True, duplicates='drop')
        self._bin_edges[col] = edges

    def _bin_numeric_column_transform(self, series: pd.Series, col: str) -> pd.DataFrame:
        """Applies pre-fitted bin edges to a numeric series, returns one-hot columns."""
        edges = self._bin_edges.get(col)
        if edges is None:
            return series.notna().astype(int).rename(f"{col}_has_value").to_frame()

        labels = [f"{col}_bin{i}" for i in range(len(edges) - 1)]
        binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        dummies = pd.get_dummies(binned, prefix=col, dtype=int)

        for label in labels:
            if label not in dummies.columns:
                dummies[label] = 0

        dummies[f"{col}_missing"] = series.isna().astype(int)
        return dummies.reindex(sorted(dummies.columns), axis=1)

    # ------------------------------------------------------------------
    # FEATURES: low-cardinality categoricals → one-hot
    # ------------------------------------------------------------------

    def _ohe_fit(self, series: pd.Series, col: str):
        """Records the unique categories seen during training."""
        self._ohe_categories[col] = series.dropna().unique().tolist()

    def _ohe_transform(self, series: pd.Series, col: str) -> pd.DataFrame:
        """One-hot encodes a categorical column using training-time categories."""
        categories = self._ohe_categories.get(col, [])
        filtered = series.where(series.isin(categories), other=np.nan)
        dummies = pd.get_dummies(filtered, prefix=col, dtype=int)
        for cat in categories:
            expected = f"{col}_{cat}"
            if expected not in dummies.columns:
                dummies[expected] = 0
        dummies[f"{col}_missing"] = series.isna().astype(int)
        return dummies

    # ------------------------------------------------------------------
    # FEATURES: list-like columns → frequency-based binary flags
    # ------------------------------------------------------------------

    def _listcol_fit(self, series: pd.Series, col: str):
        """
        Counts how often each value appears across all rows,
        then records the top-N most frequent values as binary features.
        """
        value_counts = defaultdict(int)
        for val in series.dropna():
            if str(val).strip().lower() in ('nan', 'unset', ''):
                continue
            for item in str(val).split(','):
                item = item.strip()
                if item:
                    value_counts[item] += 1

        top_values = sorted(value_counts, key=value_counts.get, reverse=True)[:self.top_n_per_list_col]
        self._top_values_per_col[col] = top_values
        self._vprint(f"  '{col}': top {len(top_values)} values selected (e.g. {top_values[:3]})")

    def _listcol_transform(self, series: pd.Series, col: str) -> pd.DataFrame:
        """Creates one binary column per top-N value: 1 if the value appears in that row."""
        top_values = self._top_values_per_col.get(col, [])
        result = {}
        for val in top_values:
            safe_name = re.sub(r'[^a-z0-9_]', '_', val.lower())
            col_name = f"{col}__{safe_name}"
            result[col_name] = series.apply(
                lambda x: 1 if isinstance(x, str) and val in x else 0
            )
        return pd.DataFrame(result, index=series.index)

    # ------------------------------------------------------------------
    # FEATURES: text columns → hand-crafted binary/count features
    # ------------------------------------------------------------------

    def _text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts hand-crafted features from 'overview' and 'tagline'.
        No embeddings or TF-IDF — only binary keyword flags and simple statistics.
        """
        features = pd.DataFrame(index=df.index)

        for text_col in ['overview', 'tagline']:
            if text_col not in df.columns:
                continue

            text = df[text_col].fillna('').astype(str).str.lower()

            features[f"{text_col}_has_text"] = (text.str.strip() != '').astype(int)

            word_counts = text.str.split().str.len().fillna(0)
            features[f"{text_col}_short"]  = (word_counts < 10).astype(int)
            features[f"{text_col}_medium"] = ((word_counts >= 10) & (word_counts < 40)).astype(int)
            features[f"{text_col}_long"]   = (word_counts >= 40).astype(int)

            for genre, keywords in self.GENRE_KEYWORDS.items():
                pattern = '|'.join(keywords)
                features[f"{text_col}_kw_{genre}"] = text.str.contains(pattern, regex=True).astype(int)

        if 'adult' in df.columns:
            features['adult'] = df['adult'].astype(int)

        return features

    # ------------------------------------------------------------------
    # MAIN FIT / TRANSFORM
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        """
        Learns all encoding parameters from the training DataFrame.
        Must be called before transform(). Does NOT transform the data.
        """
        self._vprint("\n" + "=" * 80)
        self._vprint("FITTING FEATURE ENGINEER ON TRAINING DATA")
        self._vprint("=" * 80)

        # Target: fit the three genre slot encoders
        self._vprint("\n→ Fitting genre slot encoders...")
        self.fit_target(df['genres'])

        # Numeric → bins
        self._vprint("\n→ Fitting numeric bins...")
        for col in [c for c in self.NUMERIC_TO_BIN if c in df.columns]:
            self._bin_numeric_column_fit(df[col], col)
        self._vprint(f"  ✓ {len(self.NUMERIC_TO_BIN)} numeric columns fitted")

        # Low-cardinality categoricals → OHE
        self._vprint("\n→ Fitting categorical encoders...")
        for col in [c for c in self.LOW_CARDINALITY_COLS if c in df.columns]:
            self._ohe_fit(df[col], col)
        self._vprint(f"  ✓ {len(self.LOW_CARDINALITY_COLS)} categorical columns fitted")

        # List-like → top-N frequency flags
        self._vprint("\n→ Fitting list-column top-N selectors...")
        for col in [c for c in self.LIST_LIKE_COLS if c in df.columns]:
            self._listcol_fit(df[col], col)

        self._fitted = True
        self._feature_columns = None
        self._vprint("\n✓ FeatureEngineer fitted.\n")

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: assembles the raw binary feature matrix from all encoders."""
        parts = []

        for col in [c for c in self.NUMERIC_TO_BIN if c in df.columns]:
            parts.append(self._bin_numeric_column_transform(df[col], col))

        for col in [c for c in self.LOW_CARDINALITY_COLS if c in df.columns]:
            parts.append(self._ohe_transform(df[col], col))

        for col in [c for c in self.LIST_LIKE_COLS if c in df.columns]:
            parts.append(self._listcol_transform(df[col], col))

        parts.append(self._text_features(df))

        return pd.concat(parts, axis=1).fillna(0).astype(int)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Transforms a DataFrame into (X, y) where:
        - X: binary feature matrix ready for BernoulliNB
        - y: DataFrame with columns genre_1, genre_2, genre_3 (integer-encoded)
             or None if 'genres' column is absent

        The feature schema is locked after fit_transform — test data is
        reindexed to match exactly, with missing columns filled as 0.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X = self._build_X(df)

        if self._feature_columns is not None:
            X = X.reindex(columns=self._feature_columns, fill_value=0)

        y = None
        if 'genres' in df.columns:
            y = self.transform_target(df['genres'])

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits on df, transforms it, and locks in the training column schema.
        Always call this on training data — never on test data.
        """
        self.fit(df)
        X = self._build_X(df)

        self._feature_columns = X.columns.tolist()
        self._vprint(f"✓ Feature schema locked: {len(self._feature_columns)} columns")

        y = self.transform_target(df['genres'])
        return X, y


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_model(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                alpha: float = 1.0) -> MultiOutputClassifier:
    """
    Trains a Binary Relevance Naive Bayes model for the three genre slots.
    One BernoulliNB classifier is trained independently per slot
    (genre_1, genre_2, genre_3).

    Args:
        X_train: Binary feature matrix
        y_train: Integer target DataFrame with columns genre_1, genre_2, genre_3
        alpha:   Laplace smoothing for BernoulliNB (default 1.0)

    Returns:
        Fitted MultiOutputClassifier wrapping BernoulliNB
    """
    model = MultiOutputClassifier(BernoulliNB(alpha=alpha))
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: MultiOutputClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   engineer: 'FeatureEngineer') -> pd.DataFrame:
    """
    Evaluates the three-slot genre model with per-slot accuracy and F1,
    plus two combined metrics:
      - Exact match: all three predicted slots match the true slots
      - Partial match: at least one predicted genre appears in the true genre set

    Args:
        model:    Fitted MultiOutputClassifier
        X_test:   Binary feature matrix
        y_test:   True integer target DataFrame (genre_1, genre_2, genre_3)
        engineer: Fitted FeatureEngineer (used to decode integer labels back to strings)

    Returns:
        DataFrame with per-slot classification report
    """
    y_pred_array = np.array(model.predict(X_test))  # shape (n_samples, 3)
    y_pred = pd.DataFrame(
        y_pred_array,
        columns=['genre_1', 'genre_2', 'genre_3'],
        index=y_test.index
    )

    genre_classes = engineer.get_genre_classes()

    print(f"\n{'='*60}")

    slot_reports = {}
    for slot in ['genre_1', 'genre_2', 'genre_3']:
        le = engineer._label_encoders[slot]
        classes = le.classes_

        true_labels = le.inverse_transform(y_test[slot])
        pred_labels = le.inverse_transform(y_pred[slot])

        acc = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

        print(f"  {slot}: accuracy={acc:.4f}  macro_F1={f1_macro:.4f}  weighted_F1={f1_weighted:.4f}")

        report = classification_report(
            true_labels, pred_labels,
            output_dict=True, zero_division=0
        )
        slot_reports[slot] = pd.DataFrame(report).T

    # ------------------------------------------------------------------
    # Combined metrics
    # ------------------------------------------------------------------
    # Decode all slots back to genre strings for comparison
    true_decoded = pd.DataFrame({
        slot: engineer._label_encoders[slot].inverse_transform(y_test[slot])
        for slot in ['genre_1', 'genre_2', 'genre_3']
    }, index=y_test.index)

    pred_decoded = pd.DataFrame({
        slot: engineer._label_encoders[slot].inverse_transform(y_pred[slot])
        for slot in ['genre_1', 'genre_2', 'genre_3']
    }, index=y_test.index)

    # Exact match: the full triplet is identical
    exact_match = (true_decoded.values == pred_decoded.values).all(axis=1).mean()

    # Partial match: at least one predicted genre (excluding 'none') appears
    # anywhere in the true genre set for that movie
    none_label = engineer.NONE_LABEL
    partial_matches = []
    for i in range(len(true_decoded)):
        true_set  = set(true_decoded.iloc[i]) - {none_label}
        pred_set  = set(pred_decoded.iloc[i]) - {none_label}
        partial_matches.append(len(true_set & pred_set) > 0)
    partial_match = np.mean(partial_matches)

    print(f"\n  Exact match  (all 3 slots correct): {exact_match:.4f}")
    print(f"  Partial match (≥1 genre correct):   {partial_match:.4f}")
    print(f"{'='*60}\n")

    return slot_reports