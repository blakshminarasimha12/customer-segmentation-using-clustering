import os
import sys
import math
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
np.random.seed(42)


@dataclass
class ProjectConfig:
    """Central configuration for the project."""

    random_state: int = 42
    default_k: int = 4
    min_k: int = 2
    max_k: int = 10
    sample_size: int = 300

    root_dir: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    report_dir: Path = Path("outputs/reports")

    input_file: str = "customers.csv"
    cleaned_file: str = "cleaned_customers.csv"
    segmented_file: str = "segmented_customers.csv"
    summary_file: str = "cluster_summary.csv"
    recommendations_file: str = "cluster_recommendations.csv"

    def prepare_directories(self) -> None:
        """Create all directories required by the project."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


CONFIG = ProjectConfig()
CONFIG.prepare_directories()

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
]

CATEGORICAL_FEATURES = [
    "Gender",
]

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Age",
    "Gender",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
]


def print_header(title: str) -> None:
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a smaller formatted heading."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def safe_round(value, digits: int = 2):
    """Round numeric values safely."""
    try:
        return round(float(value), digits)
    except Exception:
        return value


def show_dataframe_info(df: pd.DataFrame) -> None:
    """Display useful dataset information."""
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


def save_text(path: Path, text: str) -> None:
    """Save plain text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=4, default=str),
        encoding="utf-8",
    )

def generate_customer_dataset(
    n_customers: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic customer dataset.

    The generated data intentionally contains several behavioral
    patterns so that clustering produces useful segments.
    """

    rng = np.random.default_rng(random_state)

    customer_ids = np.arange(1001, 1001 + n_customers)

    ages = rng.integers(
        low=18,
        high=61,
        size=n_customers,
    )

    genders = rng.choice(
        ["Male", "Female"],
        size=n_customers,
        p=[0.52, 0.48],
    )

    annual_income = np.clip(
        rng.normal(
            loc=60000,
            scale=22000,
            size=n_customers,
        ),
        15000,
        150000,
    ).round().astype(int)

    spending_score = (
        48
        + 0.00025 * (annual_income - 60000)
        - 0.35 * (ages - 35)
        + rng.normal(0, 15, n_customers)
    )

    spending_score = np.clip(
        spending_score,
        1,
        100,
    ).round().astype(int)

    purchase_frequency = (
        3
        + spending_score / 12
        + rng.normal(0, 2.5, n_customers)
    )

    purchase_frequency = np.clip(
        purchase_frequency,
        1,
        30,
    ).round().astype(int)

    df = pd.DataFrame(
        {
            "Customer_ID": customer_ids,
            "Age": ages,
            "Gender": genders,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score,
            "Purchase_Frequency": purchase_frequency,
        }
    )

    return df


def create_demo_dataset(config: ProjectConfig = CONFIG) -> Path:
    """Create the demonstration dataset if it does not exist."""
    config.prepare_directories()

    destination = config.data_dir / config.input_file

    if destination.exists():
        return destination

    df = generate_customer_dataset(
        n_customers=config.sample_size,
        random_state=config.random_state,
    )

    df.to_csv(
        destination,
        index=False,
    )

    return destination

def load_customer_data(
    path: Path,
) -> pd.DataFrame:
    """Load customer data from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    return df


def ensure_dataset_exists(
    config: ProjectConfig = CONFIG,
) -> Path:
    """Ensure that a usable demo dataset exists."""
    path = config.data_dir / config.input_file

    if not path.exists():
        print("Dataset was not found.")
        print("Creating a demonstration dataset...")
        path = create_demo_dataset(config)

    return path



def validate_required_columns(
    df: pd.DataFrame,
) -> None:
    """Validate that all required columns exist."""
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def validate_numeric_columns(
    df: pd.DataFrame,
) -> None:
    """Check that required numeric fields can be interpreted numerically."""
    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_count = converted.isna().sum()

        if invalid_count > 0:
            print(
                f"Warning: {column} has "
                f"{invalid_count} non-numeric values."
            )


def validate_ranges(
    df: pd.DataFrame,
) -> Dict[str, int]:
    """Return counts of values outside expected business ranges."""
    results = {}

    if "Age" in df.columns:
        results["invalid_age"] = int(
            ((df["Age"] < 0) | (df["Age"] > 120)).sum()
        )

    if "Spending_Score" in df.columns:
        results["invalid_spending_score"] = int(
            (
                (df["Spending_Score"] < 1)
                | (df["Spending_Score"] > 100)
            ).sum()
        )

    if "Annual_Income" in df.columns:
        results["invalid_income"] = int(
            (df["Annual_Income"] < 0).sum()
        )

    if "Purchase_Frequency" in df.columns:
        results["invalid_frequency"] = int(
            (df["Purchase_Frequency"] < 0).sum()
        )

    return results


def run_validation(
    df: pd.DataFrame,
) -> Dict[str, object]:
    """Run all validation checks."""
    validate_required_columns(df)
    validate_numeric_columns(df)

    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "range_checks": validate_ranges(df),
    }



def clean_customer_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and standardize customer data."""

    cleaned = df.copy()
    cleaned.columns = [
        str(column).strip().replace(" ", "_")
        for column in cleaned.columns
    ]


    cleaned = cleaned.drop_duplicates()


    for column in NUMERIC_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    if "Gender" in cleaned.columns:
        cleaned["Gender"] = (
            cleaned["Gender"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        cleaned["Gender"] = cleaned["Gender"].replace(
            {
                "M": "Male",
                "F": "Female",
                "Man": "Male",
                "Woman": "Female",
            }
        )

    if "Age" in cleaned.columns:
        cleaned.loc[
            (cleaned["Age"] < 0)
            | (cleaned["Age"] > 120),
            "Age",
        ] = np.nan

    if "Annual_Income" in cleaned.columns:
        cleaned.loc[
            cleaned["Annual_Income"] < 0,
            "Annual_Income",
        ] = np.nan

    if "Spending_Score" in cleaned.columns:
        cleaned.loc[
            (cleaned["Spending_Score"] < 1)
            | (cleaned["Spending_Score"] > 100),
            "Spending_Score",
        ] = np.nan

    if "Purchase_Frequency" in cleaned.columns:
        cleaned.loc[
            cleaned["Purchase_Frequency"] < 0,
            "Purchase_Frequency",
        ] = np.nan

    for column in NUMERIC_FEATURES:
        if column in cleaned.columns:
            median_value = cleaned[column].median()

            cleaned[column] = cleaned[column].fillna(
                median_value
            )

    if "Gender" in cleaned.columns:
        mode = cleaned["Gender"].mode()

        if len(mode) > 0:
            cleaned["Gender"] = cleaned["Gender"].fillna(
                mode.iloc[0]
            )
        else:
            cleaned["Gender"] = cleaned["Gender"].fillna(
                "Unknown"
            )

    return cleaned.reset_index(drop=True)


def save_cleaned_data(
    df: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> Path:
    """Save cleaned data."""
    path = config.output_dir / config.cleaned_file
    df.to_csv(path, index=False)
    return path


def dataset_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return descriptive statistics."""
    return df.describe(include="all").transpose()


def missing_value_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a missing value report."""
    result = pd.DataFrame(
        {
            "Missing_Count": df.isna().sum(),
            "Missing_Percentage": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    return result.sort_values(
        "Missing_Count",
        ascending=False,
    )


def numerical_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return numerical summary for project features."""
    available = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    return (
        df[available]
        .describe()
        .round(2)
    )


def gender_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return customer count by gender."""
    if "Gender" not in df.columns:
        return pd.DataFrame()

    return (
        df["Gender"]
        .value_counts()
        .rename_axis("Gender")
        .reset_index(name="Customer_Count")
    )


def customer_value_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add derived business features.

    These features are used for analysis and not necessarily
    for the core K-Means model.
    """
    result = df.copy()

    result["Estimated_Annual_Purchases"] = (
        result["Purchase_Frequency"] * 12
    )

    result["Income_Per_Purchase"] = (
        result["Annual_Income"]
        / result["Purchase_Frequency"].replace(0, np.nan)
    )

    result["Income_Per_Purchase"] = (
        result["Income_Per_Purchase"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    result["Spending_Level"] = pd.cut(
        result["Spending_Score"],
        bins=[0, 33, 66, 100],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
    )

    result["Income_Level"] = pd.qcut(
        result["Annual_Income"],
        q=3,
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        duplicates="drop",
    )

    return result



def encode_gender(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Encode gender into a numerical column."""
    result = df.copy()

    if "Gender" not in result.columns:
        return result

    mapping = {
        "Male": 0,
        "Female": 1,
    }

    result["Gender_Code"] = (
        result["Gender"]
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )

    return result


def prepare_model_features(
    df: pd.DataFrame,
    include_gender: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare the feature matrix for clustering.

    By default, Gender is not used because customer behavior
    features are more directly related to segmentation.
    """
    prepared = encode_gender(df)

    features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature in prepared.columns
    ]

    if include_gender and "Gender_Code" in prepared.columns:
        features.append("Gender_Code")

    X = prepared[features].copy()

    return X, features


class FeatureScaler:
    """Wrapper around StandardScaler for reusable preprocessing."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
    ):
        self.feature_names = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(
        self,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)


def scale_features(
    X: pd.DataFrame,
) -> Tuple[np.ndarray, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def calculate_wcss(
    X_scaled: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict[int, float]:
    """Calculate WCSS for multiple K values."""
    results = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        model.fit(X_scaled)

        results[k] = float(model.inertia_)

    return results


def elbow_dataframe(
    wcss: Dict[int, float],
) -> pd.DataFrame:
    """Convert WCSS dictionary to a DataFrame."""
    return pd.DataFrame(
        {
            "K": list(wcss.keys()),
            "WCSS": list(wcss.values()),
        }
    )


def detect_elbow(
    wcss: Dict[int, float],
) -> int:
    """
    Estimate the elbow using maximum distance from
    the line joining the first and last points.
    """
    ks = np.array(list(wcss.keys()), dtype=float)
    values = np.array(list(wcss.values()), dtype=float)

    x1 = ks[0]
    y1 = values[0]
    x2 = ks[-1]
    y2 = values[-1]

    distances = []

    denominator = math.sqrt(
        (y2 - y1) ** 2
        + (x2 - x1) ** 2
    )

    if denominator == 0:
        return int(ks[0])

    for x, y in zip(ks, values):
        numerator = abs(
            (y2 - y1) * x
            - (x2 - x1) * y
            + x2 * y1
            - y2 * x1
        )

        distances.append(
            numerator / denominator
        )

    index = int(np.argmax(distances))

    return int(ks[index])


def plot_elbow(
    wcss: Dict[int, float],
    output_path: Path,
) -> None:
    """Save the Elbow Method graph."""
    ks = list(wcss.keys())
    values = list(wcss.values())

    plt.figure(figsize=(9, 6))

    plt.plot(
        ks,
        values,
        marker="o",
        linewidth=2,
    )

    plt.title(
        "Elbow Method for Optimal Number of Clusters"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "WCSS / Inertia"
    )

    plt.xticks(ks)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()



def calculate_silhouette_scores(
    X_scaled: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict[int, float]:
    """Calculate silhouette score for multiple K values."""
    results = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
        )

        results[k] = float(score)

    return results


def best_silhouette_k(
    scores: Dict[int, float],
) -> int:
    """Return K with the highest silhouette score."""
    return max(
        scores,
        key=scores.get,
    )


def plot_silhouette_scores(
    scores: Dict[int, float],
    output_path: Path,
) -> None:
    """Save silhouette score chart."""
    ks = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(9, 6))

    plt.plot(
        ks,
        values,
        marker="o",
        linewidth=2,
    )

    plt.title(
        "Silhouette Score by Number of Clusters"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.xticks(ks)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


class CustomerKMeansModel:
    """Reusable K-Means customer segmentation model."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self.feature_names: List[str] = []

    def fit(
        self,
        X_scaled: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Fit K-Means."""
        self.model.fit(X_scaled)

        if feature_names is not None:
            self.feature_names = list(
                feature_names
            )

        return self

    def predict(
        self,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        """Predict cluster labels."""
        return self.model.predict(
            X_scaled
        )

    def fit_predict(
        self,
        X_scaled: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(
            X_scaled,
            feature_names,
        )

        return self.model.labels_

    @property
    def centers(self) -> np.ndarray:
        """Return scaled cluster centers."""
        return self.model.cluster_centers_

    @property
    def inertia(self) -> float:
        """Return model WCSS."""
        return float(
            self.model.inertia_
        )


def assign_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Add human-readable cluster IDs to the dataset."""
    result = df.copy()


    result["Cluster"] = labels + 1

    return result


def cluster_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return customer count by cluster."""
    counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .rename("Customer_Count")
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Customer_Count",
    ]

    counts["Percentage"] = (
        counts["Customer_Count"]
        / counts["Customer_Count"].sum()
        * 100
    ).round(2)

    return counts


def profile_clusters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average characteristics for every cluster.
    """
    aggregation = {}

    for column in NUMERIC_FEATURES:
        if column in df.columns:
            aggregation[column] = "mean"

    aggregation["Customer_ID"] = "count"

    summary = (
        df.groupby("Cluster")
        .agg(aggregation)
        .round(2)
    )

    if "Customer_ID" in summary.columns:
        summary = summary.rename(
            columns={
                "Customer_ID": "Customer_Count"
            }
        )

    if "Customer_Count" in summary.columns:
        total = summary["Customer_Count"].sum()

        summary["Cluster_Percentage"] = (
            summary["Customer_Count"]
            / total
            * 100
        ).round(2)

    return summary


def add_relative_scores(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add normalized relative indicators to cluster profiles."""
    result = summary.copy()

    for column in [
        "Annual_Income",
        "Spending_Score",
        "Purchase_Frequency",
    ]:
        if column in result.columns:
            minimum = result[column].min()
            maximum = result[column].max()

            if maximum != minimum:
                result[
                    f"{column}_Relative"
                ] = (
                    (result[column] - minimum)
                    / (maximum - minimum)
                ).round(3)
            else:
                result[
                    f"{column}_Relative"
                ] = 0.5

    return result


def get_cluster_centers_original_scale(
    model: CustomerKMeansModel,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Convert K-Means centers back to original units."""
    centers_scaled = model.centers

    centers_original = scaler.inverse_transform(
        centers_scaled
    )

    return pd.DataFrame(
        centers_original,
        columns=model.feature_names,
        index=[
            i + 1
            for i in range(
                len(centers_original)
            )
        ],
    )



def segment_name_from_profile(
    income: float,
    spending: float,
    frequency: float,
    income_median: float,
    spending_median: float,
    frequency_median: float,
) -> str:
    """
    Assign a descriptive business name based on
    relative income and spending behavior.
    """

    high_income = income >= income_median
    high_spending = spending >= spending_median
    high_frequency = frequency >= frequency_median

    if high_income and high_spending and high_frequency:
        return "High-Value Loyal Customers"

    if high_income and high_spending:
        return "Premium Customers"

    if high_income and not high_spending:
        return "Affluent Low-Engagement Customers"

    if not high_income and high_spending and high_frequency:
        return "Frequent Budget Customers"

    if not high_income and high_spending:
        return "Potential Growth Customers"

    if not high_income and not high_spending and not high_frequency:
        return "Budget / Low-Engagement Customers"

    if not high_income and not high_spending:
        return "Moderate Budget Customers"

    return "Moderate Customers"


def name_clusters(
    summary: pd.DataFrame,
) -> Dict[int, str]:
    """Create descriptive names for clusters."""
    if summary.empty:
        return {}

    income_median = summary[
        "Annual_Income"
    ].median()

    spending_median = summary[
        "Spending_Score"
    ].median()

    frequency_median = summary[
        "Purchase_Frequency"
    ].median()

    names = {}

    for cluster, row in summary.iterrows():
        names[int(cluster)] = (
            segment_name_from_profile(
                income=row["Annual_Income"],
                spending=row["Spending_Score"],
                frequency=row[
                    "Purchase_Frequency"
                ],
                income_median=income_median,
                spending_median=spending_median,
                frequency_median=frequency_median,
            )
        )

    return names


def apply_segment_names(
    df: pd.DataFrame,
    names: Dict[int, str],
) -> pd.DataFrame:
    """Add segment names to customer records."""
    result = df.copy()

    result["Segment_Name"] = (
        result["Cluster"]
        .map(names)
        .fillna("Unclassified")
    )

    return result



def recommendation_for_segment(
    segment_name: str,
    average_income: float,
    average_spending: float,
    average_frequency: float,
) -> Dict[str, str]:
    """Create business recommendations for a segment."""

    if "High-Value" in segment_name:
        strategy = (
            "Retain and reward loyal customers with "
            "VIP benefits, early access, and personalized offers."
        )
        campaign = (
            "VIP loyalty program, premium bundles, "
            "early-access promotions."
        )
        priority = "Very High"

    elif "Premium" in segment_name:
        strategy = (
            "Use premium positioning and personalized "
            "cross-selling to increase customer lifetime value."
        )
        campaign = (
            "Premium products, exclusive offers, "
            "personalized recommendations."
        )
        priority = "High"

    elif "Affluent" in segment_name:
        strategy = (
            "Increase engagement through personalized "
            "recommendations and targeted incentives."
        )
        campaign = (
            "Personalized product recommendations, "
            "limited-time incentives."
        )
        priority = "High"

    elif "Potential Growth" in segment_name:
        strategy = (
            "Convert promising customers into frequent "
            "buyers using discounts and loyalty incentives."
        )
        campaign = (
            "Starter loyalty rewards, coupons, "
            "bundle offers."
        )
        priority = "High"

    elif "Frequent Budget" in segment_name:
        strategy = (
            "Maintain frequency while increasing basket "
            "value through bundles and cross-selling."
        )
        campaign = (
            "Value bundles, buy-more-save-more offers."
        )
        priority = "Medium"

    elif "Budget" in segment_name:
        strategy = (
            "Focus on affordability and avoid excessive "
            "marketing spend."
        )
        campaign = (
            "Discounts, essential products, "
            "low-cost loyalty incentives."
        )
        priority = "Medium"

    else:
        strategy = (
            "Test personalized offers and monitor "
            "changes in spending behavior."
        )
        campaign = (
            "A/B test coupons, recommendations, "
            "and seasonal promotions."
        )
        priority = "Medium"

    return {
        "Strategy": strategy,
        "Suggested_Campaign": campaign,
        "Priority": priority,
    }


def build_recommendation_table(
    summary: pd.DataFrame,
    names: Dict[int, str],
) -> pd.DataFrame:
    """Build a recommendation table for all clusters."""

    records = []

    for cluster, row in summary.iterrows():

        segment_name = names.get(
            int(cluster),
            "Unclassified",
        )

        recommendation = recommendation_for_segment(
            segment_name=segment_name,
            average_income=row["Annual_Income"],
            average_spending=row["Spending_Score"],
            average_frequency=row[
                "Purchase_Frequency"
            ],
        )

        record = {
            "Cluster": int(cluster),
            "Segment_Name": segment_name,
            "Average_Income": row[
                "Annual_Income"
            ],
            "Average_Spending_Score": row[
                "Spending_Score"
            ],
            "Average_Purchase_Frequency": row[
                "Purchase_Frequency"
            ],
        }

        record.update(recommendation)

        records.append(record)

    return pd.DataFrame(records)



def setup_plot_style() -> None:
    """Set a clean plotting style."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 10,
        }
    )

    if sns is not None:
        sns.set_theme(
            style="whitegrid"
        )


def plot_income_spending(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot annual income versus spending score."""

    plt.figure(figsize=(10, 7))

    if sns is not None:
        sns.scatterplot(
            data=df,
            x="Annual_Income",
            y="Spending_Score",
            hue="Cluster",
            style="Cluster",
            palette="viridis",
            s=90,
        )
    else:
        for cluster in sorted(
            df["Cluster"].unique()
        ):
            subset = df[
                df["Cluster"] == cluster
            ]

            plt.scatter(
                subset["Annual_Income"],
                subset["Spending_Score"],
                label=f"Cluster {cluster}",
            )

    plt.title(
        "Customer Segmentation: Income vs Spending Score"
    )

    plt.xlabel(
        "Annual Income (₹)"
    )

    plt.ylabel(
        "Spending Score (1-100)"
    )

    plt.legend(
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_age_spending(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot age versus spending score."""

    plt.figure(figsize=(10, 7))

    if sns is not None:
        sns.scatterplot(
            data=df,
            x="Age",
            y="Spending_Score",
            hue="Cluster",
            palette="viridis",
            s=80,
        )

    plt.title(
        "Customer Segmentation: Age vs Spending Score"
    )

    plt.xlabel("Age")
    plt.ylabel("Spending Score")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_income_frequency(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot income versus purchase frequency."""

    plt.figure(figsize=(10, 7))

    if sns is not None:
        sns.scatterplot(
            data=df,
            x="Annual_Income",
            y="Purchase_Frequency",
            hue="Cluster",
            palette="viridis",
            s=80,
        )

    plt.title(
        "Annual Income vs Purchase Frequency"
    )

    plt.xlabel(
        "Annual Income (₹)"
    )

    plt.ylabel(
        "Purchase Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_cluster_counts(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot number of customers in each cluster."""

    counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        counts.index.astype(str),
        counts.values,
    )

    plt.title(
        "Number of Customers in Each Cluster"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Customer Count")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_gender_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot gender distribution by cluster."""

    if "Gender" not in df.columns:
        return

    cross = pd.crosstab(
        df["Cluster"],
        df["Gender"],
    )

    cross.plot(
        kind="bar",
        figsize=(10, 7),
    )

    plt.title(
        "Gender Distribution by Customer Cluster"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Customer Count")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_feature_boxplots(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a boxplot for each important numeric feature."""

    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:
            continue

        plt.figure(figsize=(10, 6))

        if sns is not None:
            sns.boxplot(
                data=df,
                x="Cluster",
                y=feature,
            )

        plt.title(
            f"{feature} Distribution by Cluster"
        )

        plt.tight_layout()

        path = (
            output_dir
            / f"boxplot_{feature}.png"
        )

        plt.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


def perform_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, PCA]:
    """Reduce scaled features to two PCA dimensions."""
    pca = PCA(
        n_components=n_components,
        random_state=42,
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    return X_pca, pca


def plot_pca_clusters(
    X_pca: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot clusters in two-dimensional PCA space."""

    plt.figure(figsize=(10, 7))

    labels_one_based = labels + 1

    for cluster in sorted(
        np.unique(labels_one_based)
    ):

        mask = (
            labels_one_based == cluster
        )

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=70,
            label=f"Cluster {cluster}",
        )

    plt.title(
        "Customer Clusters in PCA Space"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()



def calculate_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate correlation matrix."""
    available = [
        feature
        for feature in NUMERIC_FEATURES
        if feature in df.columns
    ]

    return df[
        available
    ].corr().round(3)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save feature correlation heatmap."""

    correlation = calculate_correlation(
        df
    )

    plt.figure(
        figsize=(9, 7)
    )

    if sns is not None:
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
        )
    else:
        plt.imshow(
            correlation,
            interpolation="nearest",
        )

    plt.title(
        "Customer Feature Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()



def calculate_iqr_outliers(
    series: pd.Series,
) -> pd.Series:
    """Return a Boolean mask for IQR-based outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (
        (series < lower)
        | (series > upper)
    )


def outlier_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create an outlier count report."""

    records = []

    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:
            continue

        mask = calculate_iqr_outliers(
            df[feature]
        )

        records.append(
            {
                "Feature": feature,
                "Outlier_Count": int(
                    mask.sum()
                ),
                "Outlier_Percentage": round(
                    mask.mean() * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(records)


def cap_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cap numeric outliers using IQR limits.

    This is optional because K-Means is sensitive to outliers.
    """
    result = df.copy()

    for feature in NUMERIC_FEATURES:

        if feature not in result.columns:
            continue

        q1 = result[feature].quantile(
            0.25
        )

        q3 = result[feature].quantile(
            0.75
        )

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        result[feature] = result[
            feature
        ].clip(
            lower=lower,
            upper=upper,
        )

    return result


def evaluate_model(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model: CustomerKMeansModel,
) -> Dict[str, float]:
    """Calculate clustering evaluation metrics."""

    unique_labels = np.unique(labels)

    metrics = {
        "WCSS": float(
            model.inertia
        ),
    }

    if len(unique_labels) > 1:
        metrics[
            "Silhouette_Score"
        ] = float(
            silhouette_score(
                X_scaled,
                labels,
            )
        )
    else:
        metrics[
            "Silhouette_Score"
        ] = float("nan")

    return metrics


def evaluate_multiple_k(
    X_scaled: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
) -> pd.DataFrame:
    """Evaluate several K values."""
    records = []

    for k in range(
        min_k,
        max_k + 1,
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels,
        )

        records.append(
            {
                "K": k,
                "WCSS": model.inertia_,
                "Silhouette_Score": score,
            }
        )

    return pd.DataFrame(
        records
    )


def get_cluster_customers(
    df: pd.DataFrame,
    cluster: int,
) -> pd.DataFrame:
    """Return all customers belonging to a cluster."""
    return df[
        df["Cluster"] == cluster
    ].copy()


def get_segment_customers(
    df: pd.DataFrame,
    segment_name: str,
) -> pd.DataFrame:
    """Return customers belonging to a named segment."""
    return df[
        df["Segment_Name"]
        == segment_name
    ].copy()


def top_spenders(
    df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return top customers by spending score."""
    return (
        df.sort_values(
            "Spending_Score",
            ascending=False,
        )
        .head(n)
        .copy()
    )


def top_income_customers(
    df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return customers with highest annual income."""
    return (
        df.sort_values(
            "Annual_Income",
            ascending=False,
        )
        .head(n)
        .copy()
    )


def most_frequent_customers(
    df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return customers with highest purchase frequency."""
    return (
        df.sort_values(
            "Purchase_Frequency",
            ascending=False,
        )
        .head(n)
        .copy()
    )


def predict_new_customer(
    customer: Dict[str, object],
    scaler: StandardScaler,
    model: CustomerKMeansModel,
    feature_names: List[str],
) -> int:
    """
    Predict the segment of a new customer.

    Example input:
    {
        "Age": 28,
        "Annual_Income": 85000,
        "Spending_Score": 80,
        "Purchase_Frequency": 12
    }
    """

    row = pd.DataFrame(
        [customer]
    )

    row = row[
        feature_names
    ]

    scaled = scaler.transform(
        row
    )

    cluster = model.predict(
        scaled
    )[0]

    return int(cluster + 1)


def explain_new_customer(
    customer: Dict[str, object],
    segment_number: int,
    segment_names: Dict[int, str],
) -> str:
    """Return a simple business explanation."""
    name = segment_names.get(
        segment_number,
        "Unclassified",
    )

    return (
        f"The customer is assigned to "
        f"Cluster {segment_number} "
        f"({name})."
    )


def generate_project_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    names: Dict[int, str],
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate a text report summarizing the project."""

    lines = []

    lines.append(
        "CUSTOMER SEGMENTATION PROJECT REPORT"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Total customers: {len(df)}"
    )

    lines.append(
        f"Number of clusters: "
        f"{df['Cluster'].nunique()}"
    )

    lines.append(
        f"WCSS: {metrics.get('WCSS', float('nan')):.2f}"
    )

    lines.append(
        "Silhouette Score: "
        f"{metrics.get('Silhouette_Score', float('nan')):.4f}"
    )

    lines.append("")

    lines.append(
        "CLUSTER PROFILES"
    )

    lines.append(
        "-" * 60
    )

    for cluster, row in summary.iterrows():

        name = names.get(
            int(cluster),
            "Unclassified",
        )

        lines.append(
            f"Cluster {cluster}: {name}"
        )

        lines.append(
            f"  Customers: "
            f"{int(row['Customer_Count'])}"
        )

        lines.append(
            f"  Average age: "
            f"{row['Age']:.2f}"
        )

        lines.append(
            f"  Average income: "
            f"₹{row['Annual_Income']:.2f}"
        )

        lines.append(
            f"  Average spending score: "
            f"{row['Spending_Score']:.2f}"
        )

        lines.append(
            f"  Average purchase frequency: "
            f"{row['Purchase_Frequency']:.2f}"
        )

        lines.append("")

    save_text(
        output_path,
        "\n".join(lines),
    )


def generate_json_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    names: Dict[int, str],
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate a machine-readable JSON report."""

    clusters = []

    for cluster, row in summary.iterrows():

        clusters.append(
            {
                "cluster": int(cluster),
                "segment_name": names.get(
                    int(cluster),
                    "Unclassified",
                ),
                "customer_count": int(
                    row["Customer_Count"]
                ),
                "average_age": float(
                    row["Age"]
                ),
                "average_income": float(
                    row["Annual_Income"]
                ),
                "average_spending_score": float(
                    row["Spending_Score"]
                ),
                "average_purchase_frequency": float(
                    row["Purchase_Frequency"]
                ),
            }
        )

    report = {
        "project": (
            "Customer Segmentation "
            "using K-Means Clustering"
        ),
        "total_customers": int(
            len(df)
        ),
        "cluster_count": int(
            df["Cluster"].nunique()
        ),
        "metrics": metrics,
        "clusters": clusters,
    }

    save_json(
        output_path,
        report,
    )



def export_all_results(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> Dict[str, Path]:
    """Export all major project outputs."""

    segmented_path = (
        config.output_dir
        / config.segmented_file
    )

    summary_path = (
        config.output_dir
        / config.summary_file
    )

    recommendation_path = (
        config.output_dir
        / config.recommendations_file
    )

    df.to_csv(
        segmented_path,
        index=False,
    )

    summary.to_csv(
        summary_path
    )

    recommendations.to_csv(
        recommendation_path,
        index=False,
    )

    return {
        "segmented_data": segmented_path,
        "summary": summary_path,
        "recommendations": recommendation_path,
    }


def export_statistics(
    df: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> None:
    """Export statistical analysis tables."""

    stats_path = (
        config.output_dir
        / "dataset_statistics.csv"
    )

    missing_path = (
        config.output_dir
        / "missing_value_report.csv"
    )

    outlier_path = (
        config.output_dir
        / "outlier_report.csv"
    )

    correlation_path = (
        config.output_dir
        / "correlation_matrix.csv"
    )

    dataset_statistics(
        df
    ).to_csv(
        stats_path
    )

    missing_value_report(
        df
    ).to_csv(
        missing_path
    )

    outlier_report(
        df
    ).to_csv(
        outlier_path,
        index=False,
    )

    calculate_correlation(
        df
    ).to_csv(
        correlation_path
    )


def run_customer_segmentation(
    config: ProjectConfig = CONFIG,
) -> Dict[str, object]:
    """Run the complete customer segmentation pipeline."""

    setup_plot_style()

    config.prepare_directories()

    print_header(
        "CUSTOMER SEGMENTATION USING K-MEANS"
    )


    print_header(
        "STEP 1 - DATA COLLECTION"
    )

    dataset_path = (
        ensure_dataset_exists(
            config
        )
    )

    print(
        f"Dataset path: {dataset_path}"
    )

 
    print_header(
        "STEP 2 - DATA LOADING"
    )

    raw_df = load_customer_data(
        dataset_path
    )

    show_dataframe_info(
        raw_df
    )

    print(
        "\nFirst five rows:"
    )

    print(
        raw_df.head()
    )

 

    print_header(
        "STEP 3 - DATA VALIDATION"
    )

    validation = run_validation(
        raw_df
    )

    print(
        json.dumps(
            validation,
            indent=4,
            default=str,
        )
    )

 

    print_header(
        "STEP 4 - DATA PREPROCESSING"
    )

    df = clean_customer_data(
        raw_df
    )

    cleaned_path = save_cleaned_data(
        df,
        config,
    )

    print(
        f"Cleaned dataset saved to: "
        f"{cleaned_path}"
    )


    print_header(
        "STEP 5 - FEATURE ENGINEERING"
    )

    df = customer_value_features(
        df
    )

    print(
        df[
            [
                "Customer_ID",
                "Spending_Level",
                "Income_Level",
            ]
        ].head()
    )


    print_header(
        "STEP 6 - EXPLORATORY DATA ANALYSIS"
    )

    print_subheader(
        "Numerical Summary"
    )

    print(
        numerical_summary(
            df
        )
    )

    print_subheader(
        "Gender Summary"
    )

    print(
        gender_summary(
            df
        )
    )

    export_statistics(
        df,
        config,
    )

  

    print_header(
        "STEP 7 - FEATURE SELECTION"
    )

    X, feature_names = (
        prepare_model_features(
            df,
            include_gender=False,
        )
    )

    print(
        "Features used for clustering:"
    )

    for feature in feature_names:
        print(
            f"  - {feature}"
        )


    print_header(
        "STEP 8 - FEATURE SCALING"
    )

    X_scaled, scaler = (
        scale_features(X)
    )

    print(
        f"Scaled matrix shape: "
        f"{X_scaled.shape}"
    )


    print_header(
        "STEP 9 - ELBOW METHOD"
    )

    wcss = calculate_wcss(
        X_scaled,
        config.min_k,
        config.max_k,
        config.random_state,
    )

    elbow_k = detect_elbow(
        wcss
    )

    print(
        "WCSS values:"
    )

    for k, value in wcss.items():
        print(
            f"  K={k}: "
            f"{value:.2f}"
        )

    print(
        f"\nEstimated elbow K: "
        f"{elbow_k}"
    )

    elbow_path = (
        config.figure_dir
        / "elbow_method.png"
    )

    plot_elbow(
        wcss,
        elbow_path,
    )


    print_header(
        "STEP 10 - SILHOUETTE ANALYSIS"
    )

    silhouette_scores = (
        calculate_silhouette_scores(
            X_scaled,
            config.min_k,
            config.max_k,
            config.random_state,
        )
    )

    silhouette_k = (
        best_silhouette_k(
            silhouette_scores
        )
    )

    for k, score in (
        silhouette_scores.items()
    ):
        print(
            f"  K={k}: "
            f"{score:.4f}"
        )

    print(
        f"\nBest silhouette K: "
        f"{silhouette_k}"
    )

    silhouette_path = (
        config.figure_dir
        / "silhouette_scores.png"
    )

    plot_silhouette_scores(
        silhouette_scores,
        silhouette_path,
    )


    print_header(
        "STEP 11 - SELECT NUMBER OF CLUSTERS"
    )

    # The project synopsis specifies K-Means and examples of
    # several customer groups. K=4 is retained as the default.
    final_k = config.default_k

    print(
        f"Final K selected: {final_k}"
    )



    print_header(
        "STEP 12 - TRAIN K-MEANS MODEL"
    )

    model = CustomerKMeansModel(
        n_clusters=final_k,
        random_state=config.random_state,
    )

    labels = model.fit_predict(
        X_scaled,
        feature_names,
    )

    print(
        "K-Means training completed."
    )

    print(
        f"Model inertia: "
        f"{model.inertia:.2f}"
    )



    print_header(
        "STEP 13 - ASSIGN CUSTOMER CLUSTERS"
    )

    df = assign_clusters(
        df,
        labels,
    )

    print(
        df[
            [
                "Customer_ID",
                "Age",
                "Annual_Income",
                "Spending_Score",
                "Purchase_Frequency",
                "Cluster",
            ]
        ].head(10)
    )

  

    print_header(
        "STEP 14 - CLUSTER ANALYSIS"
    )

    summary = profile_clusters(
        df
    )

    summary = add_relative_scores(
        summary
    )

    print(
        summary
    )


    print_header(
        "STEP 15 - CUSTOMER SEGMENT INTERPRETATION"
    )

    names = name_clusters(
        summary
    )

    for cluster, name in names.items():
        print(
            f"Cluster {cluster}: "
            f"{name}"
        )

    df = apply_segment_names(
        df,
        names,
    )


    print_header(
        "STEP 16 - BUSINESS RECOMMENDATIONS"
    )

    recommendations = (
        build_recommendation_table(
            summary,
            names,
        )
    )

    print(
        recommendations[
            [
                "Cluster",
                "Segment_Name",
                "Priority",
            ]
        ]
    )


    print_header(
        "STEP 17 - MODEL EVALUATION"
    )

    metrics = evaluate_model(
        X_scaled,
        labels,
        model,
    )

    for metric, value in metrics.items():
        print(
            f"{metric}: "
            f"{value:.4f}"
        )


    print_header(
        "STEP 18 - PCA VISUALIZATION"
    )

    X_pca, pca = perform_pca(
        X_scaled
    )

    print(
        "Explained variance ratio:"
    )

    print(
        pca.explained_variance_ratio_
    )

    pca_path = (
        config.figure_dir
        / "pca_clusters.png"
    )

    plot_pca_clusters(
        X_pca,
        labels,
        pca_path,
    )


    print_header(
        "STEP 19 - CREATE VISUALIZATIONS"
    )

    plot_income_spending(
        df,
        config.figure_dir
        / "income_vs_spending.png",
    )

    plot_age_spending(
        df,
        config.figure_dir
        / "age_vs_spending.png",
    )

    plot_income_frequency(
        df,
        config.figure_dir
        / "income_vs_frequency.png",
    )

    plot_cluster_counts(
        df,
        config.figure_dir
        / "cluster_counts.png",
    )

    plot_gender_distribution(
        df,
        config.figure_dir
        / "gender_distribution.png",
    )

    plot_feature_boxplots(
        df,
        config.figure_dir,
    )

    plot_correlation_heatmap(
        df,
        config.figure_dir
        / "correlation_heatmap.png",
    )

    print(
        "Visualizations created."
    )


    print_header(
        "STEP 20 - EXPORT RESULTS"
    )

    exported = export_all_results(
        df,
        summary,
        recommendations,
        config,
    )

    for name, path in exported.items():
        print(
            f"{name}: {path}"
        )


    print_header(
        "STEP 21 - GENERATE REPORTS"
    )

    report_path = (
        config.report_dir
        / "customer_segmentation_report.txt"
    )

    json_path = (
        config.report_dir
        / "customer_segmentation_report.json"
    )

    generate_project_report(
        df,
        summary,
        names,
        metrics,
        report_path,
    )

    generate_json_report(
        df,
        summary,
        names,
        metrics,
        json_path,
    )

    print(
        f"Text report: {report_path}"
    )

    print(
        f"JSON report: {json_path}"
    )


    print_header(
        "STEP 22 - NEW CUSTOMER PREDICTION"
    )

    example_customer = {
        "Age": 28,
        "Annual_Income": 85000,
        "Spending_Score": 80,
        "Purchase_Frequency": 12,
    }

    predicted_cluster = (
        predict_new_customer(
            example_customer,
            scaler,
            model,
            feature_names,
        )
    )

    explanation = (
        explain_new_customer(
            example_customer,
            predicted_cluster,
            names,
        )
    )

    print(
        "Example customer:"
    )

    print(
        example_customer
    )

    print(
        explanation
    )



    print_header(
        "PROJECT COMPLETED"
    )

    print(
        "Customer segmentation completed successfully."
    )

    print(
        f"Customers analyzed: {len(df)}"
    )

    print(
        f"Clusters created: "
        f"{df['Cluster'].nunique()}"
    )

    print(
        "Main result file: "
        f"{config.output_dir / config.segmented_file}"
    )

    return {
        "data": df,
        "summary": summary,
        "recommendations": recommendations,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "names": names,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "pca": pca,
    }




def print_usage() -> None:
    """Print command-line usage instructions."""
    print(
        """
Customer Segmentation using K-Means

Usage:
    python customer_segmentation_2500_lines.py

The program will:
    1. Create customers.csv if needed.
    2. Clean and validate the dataset.
    3. Perform exploratory analysis.
    4. Scale model features.
    5. Calculate the Elbow Method.
    6. Calculate silhouette scores.
    7. Train K-Means.
    8. Assign customer clusters.
    9. Profile each cluster.
    10. Generate business recommendations.
    11. Save graphs and reports.
"""
    )


def main() -> None:
    """Application entry point."""
    try:
        run_customer_segmentation(
            CONFIG
        )

    except FileNotFoundError as error:
        print(
            f"File error: {error}"
        )
        sys.exit(1)

    except ValueError as error:
        print(
            f"Data error: {error}"
        )
        sys.exit(1)

    except Exception as error:
        print(
            f"Unexpected error: {error}"
        )
        raise


if __name__ == "__main__":
    main()
