import numpy as np
import pandas as pd
from collections import Counter


class ReliefFHybridFeatureSelection:
    def __init__(self, k_neighbors, sampling_frequency_m, feature_types):
        self.k_neighbors = k_neighbors
        self.sampling_frequency_m = sampling_frequency_m
        self.feature_types = feature_types
        self.w = None
        self.feature_importance_ranking = None

    def _din(self, feature_idx, x_sample, y_sample, U_D_groups):
        total_sum = 0
        x_val_on_ai = x_sample[feature_idx]
        y_val_on_ai = y_sample[feature_idx]

        for Y_j_samples in U_D_groups:
            x_val_in_Yj = (Y_j_samples[:, feature_idx] == x_val_on_ai).any()
            y_val_in_Yj = (Y_j_samples[:, feature_idx] == y_val_on_ai).any()
            intersection_x_Yj = 1 if x_val_in_Yj else 0
            diff_Yj_x = len(Y_j_samples) - (1 if x_val_in_Yj else 0)
            intersection_y_Yj = 1 if y_val_in_Yj else 0
            diff_Yj_y = len(Y_j_samples) - (1 if y_val_in_Yj else 0)
            total_sum += abs(intersection_x_Yj * diff_Yj_x - intersection_y_Yj * diff_Yj_y)

        return total_sum / len(U_D_groups)

    def _calculate_diff(self, feature_idx, sample1, sample2, U_data, U_D_groups):
        if self.feature_types[feature_idx] == 'continuous':
            feature_values = U_data[:, feature_idx]
            max_ai = np.max(feature_values)
            min_ai = np.min(feature_values)
            if (max_ai - min_ai) == 0:
                return 0.0
            return abs(sample1[feature_idx] - sample2[feature_idx]) / (max_ai - min_ai)
        elif self.feature_types[feature_idx] == 'discrete':
            return self._din(feature_idx, sample1, sample2, U_D_groups)
        else:
            return 0.0

    def fit(self, S):
        U = S.drop(columns=['label']).values
        D = S['label'].values
        unique_classes = np.unique(D)
        U_D_groups = [U[D == c] for c in unique_classes]
        num_features = U.shape[1]
        self.w = np.zeros(num_features)

        for r in range(self.sampling_frequency_m):
            rand_idx = np.random.randint(0, U.shape[0])
            x = U[rand_idx]
            x_class = D[rand_idx]
            q = np.where(unique_classes == x_class)[0][0]

            distances_to_hits = []
            for hit_sample_idx, hit_sample in enumerate(U_D_groups[q]):
                if np.array_equal(hit_sample, x) and \
                        S.index[S.apply(lambda row: np.array_equal(row.drop('label').values, x), axis=1)].iloc[0] == rand_idx:
                    continue
                total_dist = sum(
                    self._calculate_diff(f_idx, x, hit_sample, U, U_D_groups) for f_idx in range(num_features))
                distances_to_hits.append((total_dist, hit_sample))
            distances_to_hits.sort(key=lambda item: item[0])
            near_hits = [item[1] for item in distances_to_hits[:self.k_neighbors]]

            near_misses_per_class = []
            for j in range(len(unique_classes)):
                if j == q:
                    continue
                distances_to_misses = []
                for miss_sample in U_D_groups[j]:
                    total_dist = sum(
                        self._calculate_diff(f_idx, x, miss_sample, U, U_D_groups) for f_idx in range(num_features))
                    distances_to_misses.append((total_dist, miss_sample))
                distances_to_misses.sort(key=lambda item: item[0])
                near_misses_per_class.append([item[1] for item in distances_to_misses[:self.k_neighbors]])

            for i in range(num_features):
                diff_hits_sum = sum(self._calculate_diff(i, x, nh, U, U_D_groups) for nh in near_hits)
                diff_miss_sum_total = 0
                class_idx_counter = 0
                for j_actual_idx, class_samples in enumerate(U_D_groups):
                    if j_actual_idx == q:
                        continue
                    if class_idx_counter < len(near_misses_per_class):
                        misses_from_class_j = near_misses_per_class[class_idx_counter]
                        term_factor = len(class_samples) / (U.shape[0] - len(U_D_groups[q]))
                        for nm in misses_from_class_j:
                            diff_miss_sum_total += term_factor * self._calculate_diff(i, x, nm, U, U_D_groups)
                        class_idx_counter += 1
                self.w[i] = self.w[i] - (1 / (self.sampling_frequency_m * self.k_neighbors)) * diff_hits_sum + \
                            (1 / (self.sampling_frequency_m * self.k_neighbors)) * diff_miss_sum_total

        self.feature_importance_ranking = np.argsort(self.w)[::-1]
        s_a = np.empty_like(self.feature_importance_ranking)
        s_a[self.feature_importance_ranking] = np.arange(1, num_features + 1)

        return self.w, s_a


class FeatureFusion:
    def __init__(self):
        pass

    def fuse(self, multiple_ranking_results, multiple_weights):
        num_features = multiple_ranking_results.shape[1]
        N_prime = multiple_ranking_results.shape[0]
        Omega = np.zeros((num_features, num_features), dtype=int)

        for j in range(N_prime):
            for feature_idx in range(num_features):
                rank = multiple_ranking_results[j, feature_idx] - 1
                Omega[feature_idx, rank] += 1

        s_prime_a = np.zeros(num_features)
        for i in range(num_features):
            max_occurrences = np.max(Omega[i, :])
            t_prime_candidates = np.where(Omega[i, :] == max_occurrences)[0]
            if len(t_prime_candidates) == 1:
                s_prime_a[i] = t_prime_candidates[0] + 1
            else:
                s_prime_a[i] = np.mean(t_prime_candidates) + 1

        avg_weights = np.mean(multiple_weights, axis=0)
        features_with_metadata = []
        for i in range(num_features):
            features_with_metadata.append((s_prime_a[i], -avg_weights[i], i))
        features_with_metadata.sort()
        final_s_prime_ranking = np.empty_like(s_prime_a)
        for rank, original_idx in enumerate(features_with_metadata):
            final_s_prime_ranking[original_idx[2]] = rank + 1

        return final_s_prime_ranking


# if __name__ == '__main__':
#     dataset_path = ""
#     df = pd.read_csv()
#     k_neighbors = 3
#     sampling_frequency_m = 100
#     feature_types = ['continuous'] * (df.shape[1] - 1)
#     reliefF_selector = ReliefFHybridFeatureSelection(k_neighbors, sampling_frequency_m, feature_types)
#     weights, ranking = reliefF_selector.fit(df)
#     fusion = FeatureFusion()
#     final_ranking = fusion.fuse(np.array([ranking]), np.array([weights]))
