import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import describe

def load_features(file_path):
    """Loads features from a .npy file."""
    try:
        features = np.load(file_path)
        print(f"Successfully loaded features from: {file_path}")
        return features
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis of Extracted Features")
    parser.add_argument("--feature_file", type=str, 
                        default="/home/balazs/Documents/code/cora/examples/extracted_features/6block_layer1_gapTrue_test_features.npy",
                        help="Path to the .npy file containing the features.")
    parser.add_argument("--plot_dir", type=str, default="../results/feature_analysis_plots",
                        help="Directory to save analysis plots.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--tsne_n_iter", type=int, default=1000, help="Number of iterations for t-SNE.")
    parser.add_argument("--tsne_subset_size", type=int, default=5000, help="Subset size for t-SNE (if full dataset is too large). Set to 0 for full dataset.")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for K-Means.")
    parser.add_argument("--num_feature_histograms", type=int, default=5, help="Number of individual feature histograms to plot.")


    args = parser.parse_args()

    # Create plot directory if it doesn't exist
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate a unique subdirectory for this run's plots
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_file_name = os.path.splitext(os.path.basename(args.feature_file))[0]
    run_plot_dir = os.path.join(args.plot_dir, f"{feature_file_name}_{run_timestamp}")
    os.makedirs(run_plot_dir, exist_ok=True)
    print(f"Saving plots to: {run_plot_dir}")

    features = load_features(args.feature_file)

    if features is None:
        return

    print(f"\n--- Basic Information ---")
    print(f"Features shape: {features.shape}")
    num_samples, num_features = features.shape
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")

    # --- Summary Statistics ---
    print(f"\n--- Summary Statistics (per feature dimension) ---")
    features_df = pd.DataFrame(features)
    summary_stats = features_df.describe()
    print(summary_stats)
    summary_stats.to_csv(os.path.join(run_plot_dir, "summary_statistics.csv"))
    print(f"Saved summary statistics to {os.path.join(run_plot_dir, 'summary_statistics.csv')}")

    # --- Overall Feature Value Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(features.flatten(), kde=True, bins=50)
    plt.title("Overall Distribution of Feature Values")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(run_plot_dir, "overall_feature_distribution.png"))
    plt.close()
    print("Saved overall feature distribution plot.")

    # --- Individual Feature Histograms ---
    print(f"\n--- Individual Feature Histograms (Sample of {min(args.num_feature_histograms, num_features)} features) ---")
    n_hist_to_plot = min(args.num_feature_histograms, num_features)
    if n_hist_to_plot > 0:
        # Plot for first N features
        fig, axes = plt.subplots(1, n_hist_to_plot, figsize=(n_hist_to_plot * 4, 4), sharey=True)
        if n_hist_to_plot == 1: # Ensure axes is iterable
            axes = [axes]
        for i in range(n_hist_to_plot):
            sns.histplot(features[:, i], kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f"Feature {i} Distribution")
            axes[i].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")
        plt.suptitle("Distribution of First Few Individual Feature Dimensions", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(run_plot_dir, "individual_feature_histograms.png"))
        plt.close()
        print(f"Saved individual feature histogram plot for first {n_hist_to_plot} features.")

    # --- Feature Correlation Matrix ---
    if num_features <= 100: # Avoid overly large correlation matrices
        print(f"\n--- Feature Correlation Matrix ---")
        correlation_matrix = np.corrcoef(features, rowvar=False) # features are (samples, features)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(run_plot_dir, "feature_correlation_heatmap.png"))
        plt.close()
        print("Saved feature correlation heatmap.")
    else:
        print(f"Skipping feature correlation matrix heatmap as number of features ({num_features}) is > 100.")

    # --- PCA Visualization ---
    print(f"\n--- PCA Visualization ---")
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features)
    print(f"Explained variance ratio (first 2 PCA components): {pca.explained_variance_ratio_}")
    print(f"Total explained variance (first 2 PCA components): {np.sum(pca.explained_variance_ratio_):.4f}")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], s=10, alpha=0.7)
    plt.title("2D PCA of Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(run_plot_dir, "pca_2d_visualization.png"))
    plt.close()
    print("Saved 2D PCA visualization.")

    # --- t-SNE Visualization ---
    print(f"\n--- t-SNE Visualization ---")
    tsne_samples = features
    if args.tsne_subset_size > 0 and num_samples > args.tsne_subset_size:
        print(f"Using a subset of {args.tsne_subset_size} samples for t-SNE for efficiency.")
        subset_indices = np.random.choice(num_samples, args.tsne_subset_size, replace=False)
        tsne_samples = features[subset_indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=args.tsne_perplexity, n_iter=args.tsne_n_iter, init='pca', learning_rate='auto')
    features_tsne = tsne.fit_transform(tsne_samples)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], s=10, alpha=0.7)
    plt.title(f"2D t-SNE of Features (Perplexity: {args.tsne_perplexity}, Samples: {tsne_samples.shape[0]})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(run_plot_dir, "tsne_2d_visualization.png"))
    plt.close()
    print("Saved 2D t-SNE visualization.")

    # --- K-Means Clustering ---
    print(f"\n--- K-Means Clustering (k={args.n_clusters}) ---")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features) # Cluster on original features

    # Visualize clusters on PCA plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=cluster_labels, palette='viridis', s=10, alpha=0.7, legend='full')
    plt.title(f"2D PCA of Features, Colored by K-Means Clusters (k={args.n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(run_plot_dir, "pca_2d_kmeans_clusters.png"))
    plt.close()
    print("Saved PCA visualization colored by K-Means clusters.")

    # Visualize clusters on t-SNE plot (using cluster labels from full data, but plotting on t-SNE subset if used)
    plt.figure(figsize=(12, 8))
    if args.tsne_subset_size > 0 and num_samples > args.tsne_subset_size:
        sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=cluster_labels[subset_indices], palette='viridis', s=10, alpha=0.7, legend='full')
    else:
        sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=cluster_labels, palette='viridis', s=10, alpha=0.7, legend='full')
    plt.title(f"2D t-SNE of Features, Colored by K-Means Clusters (k={args.n_clusters})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(run_plot_dir, "tsne_2d_kmeans_clusters.png"))
    plt.close()
    print("Saved t-SNE visualization colored by K-Means clusters.")

    # Print cluster sizes
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("\nCluster sizes:")
    print(cluster_counts)
    cluster_counts.to_csv(os.path.join(run_plot_dir, "kmeans_cluster_sizes.csv"))
    print(f"Saved K-Means cluster sizes to {os.path.join(run_plot_dir, 'kmeans_cluster_sizes.csv')}")
    
    print(f"\n--- EDA Script Finished. Plots and stats saved in {run_plot_dir} ---")

if __name__ == '__main__':
    # Need to import datetime for the timestamp in plot directory naming
    from datetime import datetime
    main() 