"""
Evaluation Metrics Suite for LEIF.
Computes Detection, Tracking, Temporal Localization IoU, Footprint Accuracy,
and Comparative Benchmark Metrics across configurations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from leif.types import EventRecord, FootprintRecord


class EvaluationMetrics:
    """
    Computes rigorous research metrics evaluating LEIF's longitudinal memory
    and footprint accuracy against ground truth or baseline systems.
    """

    @staticmethod
    def compute_temporal_iou(pred_interval: Tuple[float, float], gt_interval: Tuple[float, float]) -> float:
        """Computes 1D temporal Intersection-over-Union for event intervals."""
        p_start, p_end = pred_interval
        g_start, g_end = gt_interval

        inter_start = max(p_start, g_start)
        inter_end = min(p_end, g_end)
        inter = max(0.0, inter_end - inter_start)

        union = max(p_end, g_end) - min(p_start, g_start)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def evaluate_events(
        predicted_events: List[EventRecord],
        ground_truth_events: List[Dict[str, Any]],
        temporal_iou_threshold: float = 0.3,
    ) -> Dict[str, float]:
        """
        Evaluates precision, recall, F1, and mean temporal IoU for predicted events.
        """
        if not ground_truth_events:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "mean_temporal_iou": 1.0}

        matched_gt = set()
        matched_pred = set()
        temporal_ious = []

        for p_idx, pred in enumerate(predicted_events):
            pred_int = (pred.timestamp_start, pred.timestamp_end)
            best_iou = 0.0
            best_gt_idx = None

            for g_idx, gt in enumerate(ground_truth_events):
                if g_idx in matched_gt:
                    continue
                # Match interaction type
                if gt.get("interaction") != pred.interaction_type:
                    continue

                gt_int = (gt["t_start"], gt["t_end"])
                tiou = EvaluationMetrics.compute_temporal_iou(pred_int, gt_int)

                if tiou >= temporal_iou_threshold and tiou > best_iou:
                    best_iou = tiou
                    best_gt_idx = g_idx

            if best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                temporal_ious.append(best_iou)

        tp = len(matched_gt)
        fp = len(predicted_events) - len(matched_pred)
        fn = len(ground_truth_events) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_tiou = float(np.mean(temporal_ious)) if temporal_ious else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "mean_temporal_iou": round(mean_tiou, 3),
        }

    @staticmethod
    def evaluate_footprint_accuracy(
        predicted_footprints: Dict[str, FootprintRecord],
        ground_truth_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluates footprint frequency error, duration error, and completeness.
        """
        freq_errors = []
        dur_errors = []

        matched_entities = 0
        total_gt = len(ground_truth_stats)

        for ent, gt_stat in ground_truth_stats.items():
            gt_count = gt_stat.get("count", 1)
            gt_dur = gt_stat.get("duration", 0.0)

            if ent in predicted_footprints:
                matched_entities += 1
                pred_fp = predicted_footprints[ent]
                # Absolute percentage errors
                c_err = abs(pred_fp.interaction_count - gt_count) / max(1, gt_count)
                d_err = abs(pred_fp.total_duration - gt_dur) / max(1.0, gt_dur)
                freq_errors.append(c_err)
                dur_errors.append(d_err)
            else:
                freq_errors.append(1.0)
                dur_errors.append(1.0)

        completeness = matched_entities / total_gt if total_gt > 0 else 1.0
        mean_freq_err = float(np.mean(freq_errors)) if freq_errors else 0.0
        mean_dur_err = float(np.mean(dur_errors)) if dur_errors else 0.0

        return {
            "footprint_completeness": round(completeness, 3),
            "frequency_error_rate": round(mean_freq_err, 3),
            "duration_error_rate": round(mean_dur_err, 3),
        }
