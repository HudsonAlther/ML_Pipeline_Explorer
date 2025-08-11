"""
Data processing utilities for ML Pipeline Explorer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set dark theme for all matplotlib plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")
import joblib
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from packaging import version
import sklearn
from config.app_config import APP_CONFIG

@st.cache_data(show_spinner=False)
def fetch_dataset(ds_key, cfg):
    """Load and prepare dataset based on configuration with caching"""
    df = pd.read_csv(
        cfg['path'],
        encoding='utf-8',
        encoding_errors='ignore',
        low_memory=False  # Prevent mixed-types chunk warning
    )
    
    # Apply dataset-specific preparation
    if ds_key == "stocks":
        df = prepare_stocks(df)
    elif ds_key == "terrorism":
        df = prepare_terrorism(df)
    elif ds_key == "netflix":
        df = prepare_netflix(df)
    
    return df

@st.cache_data(show_spinner=False)
def prepare_stocks(df):
    """Prepare stocks dataset with caching"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['Name'] == 'AAPL']
    df['AboveAvg'] = (df['close'] > df['close'].mean()).astype(int)
    df = df[['open', 'high', 'low', 'volume', 'AboveAvg']].dropna()
    return df

@st.cache_data(show_spinner=False)
def prepare_terrorism(df):
    """Prepare terrorism dataset with caching"""
    df = df.copy()
    
    # Filter for successful attacks only (0 or 1)
    df = df[df['success'].isin([0, 1])]
    
    # Select only numeric columns and the target
    numeric_cols = ['nkill', 'nwound']
    target_col = 'success'
    
    # Ensure all selected columns exist
    available_cols = [col for col in numeric_cols + [target_col] if col in df.columns]
    
    if len(available_cols) < 2:  # Need at least one feature + target
        print("Warning: Not enough numeric columns in terrorism dataset")
        # Create a simple synthetic dataset
        df = pd.DataFrame({
            'nkill': np.random.randint(0, 10, 1000),
            'nwound': np.random.randint(0, 20, 1000),
            'success': np.random.randint(0, 2, 1000)
        })
    
    # Fill missing values
    df = df[available_cols].fillna(0)
    
    return df

@st.cache_data(show_spinner=False)
def prepare_netflix(df):
    """Prepare Netflix dataset with simplified features for ML and caching"""
    df = df.copy()
    
    # Filter for Movie and TV Show only
    df = df[df['type'].isin(['Movie', 'TV Show'])]
    df['is_movie'] = (df['type'] == 'Movie').astype(int)

    # Convert release_year to numeric, handling errors
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

    # Create less-leaky features
    # Title length (characters)
    df['title'] = df['title'].fillna('')
    df['title_len'] = df['title'].astype(str).str.len().astype(float)
    
    # Description length (characters)
    df['description'] = df['description'].fillna('')
    df['desc_len'] = df['description'].astype(str).str.len().astype(float)
    
    # Cast count (number of names in comma-separated list)
    df['cast'] = df['cast'].fillna('')
    df['cast_count'] = df['cast'].apply(lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(','))).astype(float)

    # Primary country (first listed)
    df['country'] = df['country'].fillna('Unknown')
    df['country_primary'] = df['country'].astype(str).str.split(',').str[0].str.strip()

    # Select final columns with reduced leakage
    final_cols = ['release_year', 'title_len', 'desc_len', 'cast_count', 'country_primary', 'is_movie']
    df = df[final_cols].dropna(subset=['release_year'])

    return df

@st.cache_data(show_spinner=False)
def create_data_exploration_plots(df, dataset_name, target_col):
    """Create comprehensive data exploration visualizations with caching"""
    plots = {}

    # Basic info with proper JSON serialization
    plots['data_info'] = {
        'shape': list(df.shape),  # Convert tuple to list
        'columns': df.columns.tolist(),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        'target_distribution': {str(k): int(v) for k, v in df[target_col].value_counts().to_dict().items()},
        'data_quality_score': float(calculate_data_quality_score(df))  # Ensure float
    }

    # Target distribution
    from visualization.viz_factory import create_matplotlib_figure
    fig = create_matplotlib_figure(8, 6, facecolor='#1a1a1a')
    ax = fig.gca()
    # Blue bars and grey background per app theme
    ax.set_facecolor(APP_CONFIG.get('matplotlib_facecolor', '#2a2a2a'))
    primary_blue = APP_CONFIG.get('colors', {}).get('primary_blue', '#005E7B')
    df[target_col].value_counts().plot(kind='bar', ax=ax, color=primary_blue)
    ax.set_title(f'Target Distribution - {dataset_name}', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Value', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plots['target_dist_plot'] = fig

    # Numeric features correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        from visualization.viz_factory import create_matplotlib_figure
        fig = create_matplotlib_figure(10, 8, facecolor='#1a1a1a')
        ax = fig.gca()
        correlation_matrix = df[numeric_cols].corr()
        # Use approved two-color gradient: mediumblue to darkred
        from matplotlib.colors import LinearSegmentedColormap
        colors_cfg = APP_CONFIG.get('colors', {})
        heatmap_cmap = LinearSegmentedColormap.from_list(
            'approved_bwr', [colors_cfg.get('mediumblue', '#005E7B'), colors_cfg.get('darkred', '#D0073A')]
        )
        sns.heatmap(
            correlation_matrix,
            annot=True,
            annot_kws={"color": "white"},
            cmap=heatmap_cmap,
            center=0,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        # Ensure colorbar text is white
        try:
            cbar = ax.collections[0].colorbar
            if cbar is not None:
                cbar.ax.yaxis.set_tick_params(color='white')
                for tick in cbar.ax.get_yticklabels():
                    tick.set_color('white')
                cbar.set_label('Correlation', color='white')
        except Exception:
            pass
        ax.set_title(f'Feature Correlation Matrix - {dataset_name}', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        plt.tight_layout()
        plots['correlation_plot'] = fig

    # Feature distributions
    if len(numeric_cols) > 1:
        from visualization.viz_factory import create_matplotlib_figure
        fig = create_matplotlib_figure(15, 10, facecolor='#1a1a1a')
        axes = fig.subplots(2, 2)
        axes = axes.ravel()
        for i, col in enumerate(numeric_cols[:4]):
            if col != target_col:
                green = APP_CONFIG.get('colors', {}).get('lightgreen', '#70B73F')
                df[col].hist(ax=axes[i], bins=30, color=green, alpha=0.7)
                axes[i].set_title(f'{col} Distribution', color='white', fontsize=12, fontweight='bold')
                axes[i].tick_params(colors='white')
                axes[i].set_xlabel(axes[i].get_xlabel(), color='white')
                axes[i].set_ylabel(axes[i].get_ylabel(), color='white')
        plt.tight_layout()
        plots['feature_dist_plot'] = fig

    return plots

def calculate_data_quality_score(df):
    """Calculate a data quality score based on missing values, duplicates, etc."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    missing_score = 1 - (missing_cells / total_cells)
    duplicate_score = 1 - (duplicate_rows / df.shape[0])
    
    return (missing_score + duplicate_score) / 2

def create_preprocessing_pipeline(df, target, ds_key, model_key):
    """Create and save a preprocessing pipeline for consistent feature encoding"""
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Create preprocessing transformers
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    else:
        categorical_transformer = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit the preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        # Get categorical feature names from encoder
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    # Save the preprocessor
    preprocessor_path = Path(f"artifacts/{ds_key}/{model_key}/preprocessor.pkl")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save feature names
    feature_names_path = Path(f"artifacts/{ds_key}/{model_key}/feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    
    return X_processed, y, preprocessor, feature_names

def load_preprocessor(ds_key, model_key):
    """Load the saved preprocessor"""
    preprocessor_path = Path(f"artifacts/{ds_key}/{model_key}/preprocessor.pkl")
    if preprocessor_path.exists():
        return joblib.load(preprocessor_path)
    return None

def load_feature_names(ds_key, model_key):
    """Load the saved feature names"""
    feature_names_path = Path(f"artifacts/{ds_key}/{model_key}/feature_names.pkl")
    if feature_names_path.exists():
        return joblib.load(feature_names_path)
    return None

def preprocess_new_data(df, preprocessor, feature_names):
    """Preprocess new data using the saved preprocessor"""
    if preprocessor is None:
        return None
    
    # Transform the data
    X_processed = preprocessor.transform(df)
    
    # Convert to DataFrame with correct feature names
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_df

def split_preprocess(df, target, test_size=0.2, random_state=42):
    """Split and preprocess data for machine learning"""
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Basic NA handling
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna('missing')

    # Convert categorical to dummy variables
    X = pd.get_dummies(X)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if y.nunique()==2 else None
    )

    return X_train, X_test, y_train, y_test, num_cols, cat_cols 