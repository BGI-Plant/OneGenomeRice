import numpy as np
import torch



def catter(data, cat_dim):
    if len(data.shape) == 2:
        return data
    else:
        if cat_dim == 1:
            return torch.reshape(data, (data.shape[0], -1))
        else:
            return data.mean(dim=1)


def predict_with_rf_models(
    embeddings: np.ndarray,
    rf_models: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions using trained Random Forest models.
    
    Args:
        embeddings: Input embeddings array
        rf_models: List of trained RandomForestClassifier models
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    preds_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    
    for rf in rf_models:
        pred_i = rf.predict(embeddings)
        prob_i = rf.predict_proba(embeddings)[:, 1]
        preds_list.append(pred_i)
        probs_list.append(prob_i)

    probs = np.column_stack(probs_list)
    preds = np.column_stack(preds_list)

    return preds, probs