from typing import Dict, List, Optional, Any


def get_label_names_from_cfg(ds_cfg: Dict[str, Any], num_class: Optional[int] = None) -> Optional[List[str]]:
    """
    Extract ordered label names from dataset config.

    Supports:
        ds_cfg["label_names"] = list
        ds_cfg["id2label"] = dict with int or str keys
    """
    if not isinstance(ds_cfg, dict):
        return None

    if "label_names" in ds_cfg and ds_cfg["label_names"] is not None:
        label_names = [str(x) for x in ds_cfg["label_names"]]

        if num_class is not None and len(label_names) != int(num_class):
            raise ValueError(
                f"len(label_names)={len(label_names)} does not match num_class={num_class}"
            )

        return label_names

    if "id2label" in ds_cfg and ds_cfg["id2label"] is not None:
        id2label = ds_cfg["id2label"]

        if not isinstance(id2label, dict):
            raise TypeError(f"ds_cfg['id2label'] should be dict, got {type(id2label)}")

        normalized = {}

        for k, v in id2label.items():
            normalized[int(k)] = str(v)

        if num_class is None:
            num_class = len(normalized)

        num_class = int(num_class)

        missing = [i for i in range(num_class) if i not in normalized]

        if len(missing) > 0:
            raise ValueError(
                f"id2label missing ids: {missing}. "
                f"Available keys: {sorted(normalized.keys())}"
            )

        return [normalized[i] for i in range(num_class)]

    return None


def get_id2label_from_cfg(ds_cfg: Dict[str, Any], num_class: Optional[int] = None) -> Optional[Dict[int, str]]:
    label_names = get_label_names_from_cfg(ds_cfg, num_class=num_class)

    if label_names is None:
        return None

    return {i: name for i, name in enumerate(label_names)}