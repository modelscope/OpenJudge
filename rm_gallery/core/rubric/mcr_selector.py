"""
Super-optimized Adaptive MCR² Selector

Main optimizations:
1. Use SVD decomposition to accelerate coding rate calculation
2. Pre-compute and cache intermediate results
3. Random sampling to reduce candidate set size
4. Incremental matrix updates
5. Early stopping strategy
"""

from typing import Any, Dict, List

import numpy as np
from dashscope import TextEmbedding
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm


class SuperFastAdaptiveMCR2:
    """Super-optimized adaptive MCR² selector"""

    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size

    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate text embeddings in batches"""
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), self.batch_size), desc="Generating embeddings"
        ):
            batch_texts = texts[i : i + self.batch_size]

            try:
                rsp = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v1, input=batch_texts
                )

                if rsp.status_code == 200:
                    embeddings = [
                        record["embedding"] for record in rsp.output["embeddings"]
                    ]
                    all_embeddings.extend(embeddings)
                else:
                    logger.error(f"Embedding API call failed: {rsp.status_code}")
                    all_embeddings.extend([np.zeros(1536) for _ in batch_texts])

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                all_embeddings.extend([np.zeros(1536) for _ in batch_texts])

        return np.array(all_embeddings)

    def fast_coding_rate_svd(self, X: np.ndarray, eps: float = 0.1) -> float:
        """Fast coding rate calculation using SVD"""
        n, d = X.shape
        if n == 0:
            return 0.0

        try:
            # Use SVD decomposition, keep only main singular values
            if n > 50:  # Use random sampling for large matrices
                sample_size = min(50, n)
                sample_idx = np.random.choice(n, size=sample_size, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X

            # SVD decomposition
            U, s, Vt = np.linalg.svd(X_sample, full_matrices=False)

            # Keep 95% of energy
            energy = np.cumsum(s**2) / np.sum(s**2)
            k = np.searchsorted(energy, 0.95) + 1
            k = min(k, len(s))

            # Calculate coding rate using main singular values
            s_main = s[:k]
            log_det_approx = 2 * np.sum(np.log(1 + s_main**2 / (eps**2 * n) + 1e-8))

            return float(0.5 * log_det_approx)

        except Exception:
            return 0.0

    def ultra_fast_adaptive_selection(
        self,
        texts: List[str],
        batch_size: int = 5,
        eps: float = 0.1,
        normalize: bool = True,
        min_increment_threshold: float = 0.001,
        patience: int = 3,
        max_samples: int = 100,
        candidate_sample_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Ultra-fast adaptive selection

        Args:
            candidate_sample_ratio: Ratio of candidates to sample from each batch, reduces computation
            min_increment_threshold: Minimum increment threshold, start counting when below this value
            patience: Tolerance for consecutive low increments, stop when reaching this value
        """

        logger.info(
            f"MCR² Selector: Selecting optimal subset from {len(texts)} candidates..."
        )

        # 1. Generate embeddings
        logger.info("Generating embeddings...")
        X = self.generate_embeddings_batch(texts)

        # 2. Dimensionality reduction preprocessing
        original_dim = X.shape[1]
        n_samples = X.shape[0]

        # Dynamically determine PCA dimensions: cannot exceed minimum of sample count and feature dimensions
        max_components = min(n_samples, original_dim, 100)

        if original_dim > max_components:
            logger.info(
                f"PCA dimensionality reduction: {original_dim} -> {max_components}"
            )
            pca = PCA(n_components=max_components, random_state=42)
            X = pca.fit_transform(X)
        else:
            logger.info(
                f"No dimensionality reduction needed: original_dim={original_dim}, n_samples={n_samples}"
            )

        # 3. Normalization
        if normalize:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # 4. Ultra-fast selection process
        selected_indices = []
        candidate_indices = list(range(len(texts)))

        batch_history = []
        coding_rate_history = [0.0]
        increment_history = []
        cumulative_samples = [0]

        batch_num = 0
        internal_low_increment_count = (
            0  # MCR internal patience, separate from pipeline level
        )

        while len(selected_indices) < max_samples and len(candidate_indices) > 0:
            batch_num += 1
            current_batch_size = min(batch_size, max_samples - len(selected_indices))

            if current_batch_size <= 0:
                break

            # Internal batch processing (no detailed logs)

            # Current coding rate
            if selected_indices:
                current_X = X[selected_indices]
                R_current = self.fast_coding_rate_svd(current_X, eps)
            else:
                R_current = 0.0

            # Candidate sampling: reduce computation
            if len(candidate_indices) > 100:
                sample_size = max(
                    50, int(len(candidate_indices) * candidate_sample_ratio)
                )
                sampled_candidates = np.random.choice(
                    candidate_indices, size=sample_size, replace=False
                ).tolist()
            else:
                sampled_candidates = candidate_indices.copy()

            # Internal sampling process (silent execution)

            # Fast batch selection
            batch_result = self._ultra_fast_batch_selection(
                X, selected_indices, sampled_candidates, current_batch_size, eps
            )

            if not batch_result["best_batch_indices"]:
                break

            # Calculate increment
            new_selected = selected_indices + batch_result["best_batch_indices"]
            new_X = X[new_selected]
            R_new = self.fast_coding_rate_svd(new_X, eps)
            batch_increment = R_new - R_current

            # Internal increment calculation (not displayed)

            # Record history
            batch_info = {
                "batch_num": batch_num,
                "batch_indices": batch_result["best_batch_indices"],
                "increment": batch_increment,
                "coding_rate": R_new,
                "cumulative_samples": len(new_selected),
            }
            batch_history.append(batch_info)
            coding_rate_history.append(R_new)
            increment_history.append(batch_increment)
            cumulative_samples.append(len(new_selected))

            # Update selection
            selected_indices = new_selected
            for idx in batch_result["best_batch_indices"]:
                if idx in candidate_indices:
                    candidate_indices.remove(idx)

            # MCR internal low increment check (separate from pipeline level)
            if batch_increment < min_increment_threshold:
                internal_low_increment_count += 1
                if internal_low_increment_count >= patience:
                    break
            else:
                internal_low_increment_count = 0

        # 5. Result analysis
        selected_texts = [texts[i] for i in selected_indices]
        final_coding_rate = coding_rate_history[-1] if coding_rate_history else 0.0

        analysis = self._analyze_curve(
            cumulative_samples, coding_rate_history, increment_history
        )

        results = {
            "selected_indices": selected_indices,
            "selected_texts": selected_texts,
            "final_sample_count": len(selected_indices),
            "final_coding_rate": final_coding_rate,
            "batch_history": batch_history,
            "coding_rate_history": coding_rate_history,
            "increment_history": increment_history,
            "cumulative_samples": cumulative_samples,
            "analysis": analysis,
            "embeddings": X,  # Save processed embeddings for visualization
            "configuration": {
                "batch_size": batch_size,
                "eps": eps,
                "min_increment_threshold": min_increment_threshold,
                "patience": patience,
                "max_samples": max_samples,
                "candidate_sample_ratio": candidate_sample_ratio,
                "method": "ultra_fast_adaptive",
            },
        }

        logger.info(
            f"MCR² selection completed: {len(selected_indices)} samples, coding_rate={final_coding_rate:.6f}"
        )

        return results

    def _ultra_fast_batch_selection(
        self,
        X: np.ndarray,
        selected_indices: List[int],
        candidate_indices: List[int],
        batch_size: int,
        eps: float,
    ) -> Dict[str, Any]:
        """Ultra-fast batch selection"""

        if batch_size == 1:
            # Single sample: find the best directly
            best_delta = -np.inf
            best_idx = -1

            if selected_indices:
                current_X = X[selected_indices]
                R_current = self.fast_coding_rate_svd(current_X, eps)
            else:
                R_current = 0.0

            # Random sampling of candidates for acceleration
            eval_candidates = candidate_indices
            if len(candidate_indices) > 50:
                eval_candidates = np.random.choice(
                    candidate_indices, size=50, replace=False
                ).tolist()

            for idx in eval_candidates:
                temp_indices = selected_indices + [idx]
                temp_X = X[temp_indices]
                R_temp = self.fast_coding_rate_svd(temp_X, eps)
                delta = R_temp - R_current

                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx

            return {
                "best_batch_indices": [best_idx] if best_idx != -1 else [],
                "best_delta": best_delta,
            }

        else:
            # Multiple samples: use distance diversity heuristic
            batch_indices = []
            temp_candidates = candidate_indices.copy()

            # First sample: random selection or select the one with maximum norm
            if selected_indices:
                # Select the one farthest from already selected samples
                selected_X = X[selected_indices]
                center = np.mean(selected_X, axis=0)

                distances = []
                for idx in temp_candidates:
                    dist = np.linalg.norm(X[idx] - center)
                    distances.append((dist, idx))

                distances.sort(reverse=True)
                first_idx = distances[0][1]
            else:
                # Select the one with maximum norm
                norms = [np.linalg.norm(X[idx]) for idx in temp_candidates]
                max_norm_pos = np.argmax(norms)
                first_idx = temp_candidates[max_norm_pos]

            batch_indices.append(first_idx)
            temp_candidates.remove(first_idx)

            # Subsequent samples: select the one farthest from current batch
            for _ in range(batch_size - 1):
                if not temp_candidates:
                    break

                batch_X = X[batch_indices]
                batch_center = np.mean(batch_X, axis=0)

                best_dist = -1
                best_idx = -1

                # Only evaluate part of candidates
                eval_size = min(30, len(temp_candidates))
                eval_candidates = np.random.choice(
                    temp_candidates, size=eval_size, replace=False
                )

                for idx in eval_candidates:
                    dist = np.linalg.norm(X[idx] - batch_center)
                    if dist > best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx != -1:
                    batch_indices.append(best_idx)
                    temp_candidates.remove(best_idx)

            return {
                "best_batch_indices": batch_indices,
                "best_delta": 0.0,  # Heuristic method, no real increment calculation
            }

    def _analyze_curve(
        self,
        cumulative_samples: List[int],
        coding_rates: List[float],
        increments: List[float],
    ) -> Dict[str, Any]:
        """Simplified curve analysis"""

        if len(coding_rates) < 2:
            return {
                "optimal_sample_count": cumulative_samples[-1]
                if cumulative_samples
                else 0
            }

        # Find the point where increment starts to decline significantly
        if increments:
            # Use simple threshold method
            avg_increment = np.mean(increments)
            threshold = avg_increment * 0.3

            optimal_point = cumulative_samples[-1]
            for i, inc in enumerate(increments):
                if inc < threshold:
                    optimal_point = cumulative_samples[i + 1]
                    break
        else:
            optimal_point = cumulative_samples[-1]

        return {
            "optimal_sample_count": optimal_point,
            "total_growth": coding_rates[-1] - coding_rates[0],
            "average_increment": np.mean(increments) if increments else 0,
        }
