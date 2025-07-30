"""
Intelligent Sampling Algorithm for Category Discovery

This module implements stratified and diversity-based sampling algorithms
to select representative subsets of tickets for category pattern analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import logging
from pathlib import Path


class IntelligentSampler:
    """
    Implements intelligent sampling strategies for ticket categorization discovery.
    
    Supports three sampling strategies:
    - stratified: Maintains temporal proportions
    - diversity: Maximizes textual diversity using TF-IDF clustering
    - hybrid: Combines temporal and diversity sampling
    """
    
    def __init__(self, strategy: str = "stratified", random_state: int = 42):
        """
        Initialize the sampler with specified strategy.
        
        Args:
            strategy: Sampling strategy ("stratified", "diversity", "hybrid")
            random_state: Random seed for reproducible results
        """
        self.strategy = strategy
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if strategy not in ["stratified", "diversity", "hybrid"]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of: stratified, diversity, hybrid")
    
    def sample_tickets(self, 
                      df: pd.DataFrame, 
                      sample_size: float = 0.15,
                      min_tickets: Optional[int] = None,
                      max_tickets: Optional[int] = None,
                      adaptive: bool = True) -> pd.DataFrame:
        """
        Sample tickets using the configured strategy.
        
        Args:
            df: DataFrame with ticket data
            sample_size: Fraction of data to sample (0.15 = 15%)
            min_tickets: Minimum number of tickets (None = auto-calculate)
            max_tickets: Maximum number of tickets (None = auto-calculate)
            adaptive: Use adaptive sizing based on dataset size
            
        Returns:
            DataFrame with sampled tickets
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if sample_size <= 0 or sample_size > 1:
            raise ValueError("sample_size must be between 0 and 1")
        
        # Calculate adaptive constraints if not provided
        total_tickets = len(df)
        if adaptive and (min_tickets is None or max_tickets is None):
            adaptive_min, adaptive_max = self._calculate_adaptive_constraints(total_tickets)
            min_tickets = min_tickets or adaptive_min
            max_tickets = max_tickets or adaptive_max
        elif min_tickets is None or max_tickets is None:
            # Fallback to PRD defaults for 19K dataset
            min_tickets = min_tickets or 500
            max_tickets = max_tickets or 1000
        
        # Calculate target sample size
        target_size = int(total_tickets * sample_size)
        
        # Apply constraints: first max, then min, ensuring logical order
        if max_tickets < min_tickets:
            self.logger.warning(f"max_tickets ({max_tickets}) is less than min_tickets ({min_tickets}). Using max_tickets.")
            target_size = min(target_size, max_tickets)
        else:
            target_size = max(min_tickets, min(target_size, max_tickets))
        
        # Ensure we don't sample more than available
        target_size = min(target_size, total_tickets)
        
        self.logger.info(f"Sampling {target_size} tickets from {total_tickets} using {self.strategy} strategy")
        
        # Apply sampling strategy
        if self.strategy == "stratified":
            return self._stratified_sampling(df, target_size)
        elif self.strategy == "diversity":
            return self._diversity_sampling(df, target_size)
        elif self.strategy == "hybrid":
            return self._hybrid_sampling(df, target_size)
    
    def _stratified_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Stratified sampling maintaining temporal proportions.
        
        Args:
            df: Input DataFrame
            target_size: Number of tickets to sample
            
        Returns:
            Sampled DataFrame
        """
        # Convert ticket_created_at to datetime
        df_copy = df.copy()
        df_copy['ticket_created_at'] = pd.to_datetime(df_copy['ticket_created_at'])
        
        # Group by month to maintain temporal distribution
        df_copy['month_year'] = df_copy['ticket_created_at'].dt.to_period('M')
        month_counts = df_copy['month_year'].value_counts().sort_index()
        
        # Calculate proportional sample size for each month
        total_tickets = len(df_copy)
        sample_fractions = target_size / total_tickets
        
        sampled_dfs = []
        for month, count in month_counts.items():
            month_data = df_copy[df_copy['month_year'] == month]
            month_sample_size = max(1, int(count * sample_fractions))
            
            if len(month_data) <= month_sample_size:
                sampled_dfs.append(month_data)
            else:
                sampled_month = month_data.sample(
                    n=month_sample_size, 
                    random_state=self.random_state
                )
                sampled_dfs.append(sampled_month)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # Adjust to exact target size if needed
        if len(result) > target_size:
            result = result.sample(n=target_size, random_state=self.random_state)
        elif len(result) < target_size:
            # Fill remaining with random samples
            remaining = target_size - len(result)
            excluded = df_copy[~df_copy.index.isin(result.index)]
            if len(excluded) > 0:
                additional = excluded.sample(
                    n=min(remaining, len(excluded)), 
                    random_state=self.random_state
                )
                result = pd.concat([result, additional], ignore_index=True)
        
        self.logger.info(f"Stratified sampling completed: {len(result)} tickets selected")
        return result.drop(columns=['month_year'])
    
    def _diversity_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Diversity sampling using TF-IDF clustering to maximize textual diversity.
        
        Args:
            df: Input DataFrame
            target_size: Number of tickets to sample
            
        Returns:
            Sampled DataFrame with maximum diversity
        """
        # Prepare text data
        text_data = df['text'].fillna('').astype(str)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Keep all words for better diversity
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
        except ValueError as e:
            self.logger.warning(f"TF-IDF failed: {e}. Falling back to random sampling.")
            return df.sample(n=target_size, random_state=self.random_state)
        
        # Perform K-means clustering
        n_clusters = min(target_size, len(df))
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        
        try:
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
        except Exception as e:
            self.logger.warning(f"K-means failed: {e}. Falling back to random sampling.")
            return df.sample(n=target_size, random_state=self.random_state)
        
        # Select representative from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find ticket closest to cluster centroid
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_vectors = tfidf_matrix[cluster_indices]
            
            # Calculate distances to centroid
            distances = cosine_distances(cluster_vectors, cluster_center.reshape(1, -1))
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
        
        # Ensure we have enough samples
        if len(selected_indices) < target_size:
            remaining = target_size - len(selected_indices)
            excluded_indices = [i for i in range(len(df)) if i not in selected_indices]
            additional_indices = np.random.choice(
                excluded_indices, 
                size=min(remaining, len(excluded_indices)), 
                replace=False
            )
            selected_indices.extend(additional_indices)
        
        result = df.iloc[selected_indices[:target_size]].copy()
        
        self.logger.info(f"Diversity sampling completed: {len(result)} tickets selected")
        return result
    
    def _hybrid_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Hybrid sampling combining temporal and diversity strategies.
        
        Args:
            df: Input DataFrame
            target_size: Number of tickets to sample
            
        Returns:
            Sampled DataFrame combining both strategies
        """
        # Split target between strategies (70% stratified, 30% diversity)
        stratified_size = int(target_size * 0.7)
        diversity_size = target_size - stratified_size
        
        # Get stratified sample
        stratified_sample = self._stratified_sampling(df, stratified_size)
        
        # Get diversity sample from remaining data
        remaining_df = df[~df.index.isin(stratified_sample.index)]
        
        if len(remaining_df) > 0:
            diversity_sample = self._diversity_sampling(remaining_df, diversity_size)
            result = pd.concat([stratified_sample, diversity_sample], ignore_index=True)
        else:
            # If no remaining data, just use stratified sample
            result = stratified_sample
        
        self.logger.info(f"Hybrid sampling completed: {len(result)} tickets selected")
        return result
    
    def _calculate_adaptive_constraints(self, total_tickets: int) -> Tuple[int, int]:
        """
        Calculate adaptive min/max constraints based on dataset size.
        
        Args:
            total_tickets: Total number of tickets in dataset
            
        Returns:
            Tuple of (min_tickets, max_tickets)
        """
        if total_tickets < 5000:
            # Small dataset: higher proportion for statistical significance
            min_tickets = max(100, int(total_tickets * 0.20))  # 20%
            max_tickets = max(500, int(total_tickets * 0.50))  # 50%
            self.logger.info(f"Small dataset detected ({total_tickets}): using 20-50% sampling")
        
        elif total_tickets < 20000:
            # Medium dataset: PRD recommended 2.6-5.2%
            min_tickets = max(500, int(total_tickets * 0.026))  # 2.6%
            max_tickets = min(1000, int(total_tickets * 0.052))  # 5.2%
            self.logger.info(f"Medium dataset detected ({total_tickets}): using 2.6-5.2% sampling")
        
        else:
            # Large dataset: lower proportion but higher absolute numbers
            min_tickets = max(1000, int(total_tickets * 0.02))   # 2%
            max_tickets = min(2000, int(total_tickets * 0.04))   # 4%
            self.logger.info(f"Large dataset detected ({total_tickets}): using 2-4% sampling")
        
        return min_tickets, max_tickets
    
    def validate_sample_representativeness(self, 
                                         original_df: pd.DataFrame, 
                                         sample_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate how representative the sample is compared to the original dataset.
        
        Args:
            original_df: Original full dataset
            sample_df: Sampled dataset
            
        Returns:
            Dictionary with representativeness metrics
        """
        metrics = {}
        
        # Temporal distribution
        original_df['date'] = pd.to_datetime(original_df['ticket_created_at'])
        sample_df['date'] = pd.to_datetime(sample_df['ticket_created_at'])
        
        original_monthly = original_df['date'].dt.to_period('M').value_counts(normalize=True)
        sample_monthly = sample_df['date'].dt.to_period('M').value_counts(normalize=True)
        
        # Calculate KL divergence for temporal distribution
        common_months = set(original_monthly.index) & set(sample_monthly.index)
        if common_months:
            orig_dist = original_monthly[list(common_months)]
            samp_dist = sample_monthly[list(common_months)]
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            orig_dist = orig_dist + epsilon
            samp_dist = samp_dist + epsilon
            
            kl_divergence = sum(samp_dist * np.log(samp_dist / orig_dist))
            metrics['temporal_kl_divergence'] = float(kl_divergence)
        
        # Text length distribution
        orig_lengths = original_df['text'].str.len()
        sample_lengths = sample_df['text'].str.len()
        
        metrics['length_mean_diff'] = abs(orig_lengths.mean() - sample_lengths.mean())
        metrics['length_std_diff'] = abs(orig_lengths.std() - sample_lengths.std())
        
        # Sender distribution
        orig_senders = original_df['sender'].value_counts(normalize=True)
        sample_senders = sample_df['sender'].value_counts(normalize=True)
        
        common_senders = set(orig_senders.index) & set(sample_senders.index)
        if common_senders:
            sender_diff = sum(abs(orig_senders[s] - sample_senders.get(s, 0)) for s in common_senders)
            metrics['sender_distribution_diff'] = float(sender_diff)
        
        # Coverage metrics
        metrics['sample_size'] = len(sample_df)
        metrics['sample_ratio'] = len(sample_df) / len(original_df)
        metrics['unique_tickets_ratio'] = sample_df['ticket_id'].nunique() / original_df['ticket_id'].nunique()
        
        self.logger.info(f"Sample validation completed: {metrics}")
        return metrics


def create_diversity_report(sampler: IntelligentSampler, 
                          original_df: pd.DataFrame, 
                          sample_df: pd.DataFrame) -> str:
    """
    Create a detailed diversity report for the sampling results.
    
    Args:
        sampler: The sampler instance used
        original_df: Original dataset
        sample_df: Sampled dataset
        
    Returns:
        Formatted report string
    """
    metrics = sampler.validate_sample_representativeness(original_df, sample_df)
    
    report = f"""
# Sampling Diversity Report

## Strategy: {sampler.strategy}

## Sample Statistics
- Total tickets sampled: {metrics['sample_size']:,}
- Sample ratio: {metrics['sample_ratio']:.1%}
- Unique tickets coverage: {metrics['unique_tickets_ratio']:.1%}

## Representativeness Metrics
- Temporal KL Divergence: {metrics.get('temporal_kl_divergence', 'N/A'):.4f}
- Length mean difference: {metrics['length_mean_diff']:.1f} characters
- Length std difference: {metrics['length_std_diff']:.1f} characters
- Sender distribution difference: {metrics.get('sender_distribution_diff', 'N/A'):.4f}

## Quality Assessment
"""
    
    # Add quality assessment
    if metrics.get('temporal_kl_divergence', float('inf')) < 0.1:
        report += "✅ Excellent temporal representativeness\n"
    elif metrics.get('temporal_kl_divergence', float('inf')) < 0.3:
        report += "⚠️ Good temporal representativeness\n"
    else:
        report += "❌ Poor temporal representativeness\n"
    
    if metrics['sample_ratio'] >= 0.026:  # 2.6% minimum from PRD
        report += "✅ Sample size meets minimum requirements\n"
    else:
        report += "❌ Sample size below minimum requirements\n"
    
    if metrics['unique_tickets_ratio'] >= 0.8:
        report += "✅ Good ticket diversity coverage\n"
    else:
        report += "⚠️ Limited ticket diversity coverage\n"
    
    return report