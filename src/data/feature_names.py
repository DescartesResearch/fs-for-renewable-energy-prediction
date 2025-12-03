def get_features_by_tags(
        feature_tags: dict,
        tags_include_all: set | list = None,
        tags_include_any: set | list = None,
        tags_exclude: set | list = None
) -> list:
    """
    Filter features by tags with support for intersection (all), union (any), and exclusion.

    Args:
        feature_tags (dict): Mapping of feature names to tag sets
        tags_include_all (set or list): Tags that must all be present
        tags_include_any (set or list): At least one of these tags must be present
        tags_exclude (set or list): Tags that must not be present

    Returns:
        list: Matching feature names
    """
    tags_include_all = set(tags_include_all or [])
    tags_include_any = set(tags_include_any or [])
    tags_exclude = set(tags_exclude or [])

    selected = []
    for feature, tags in feature_tags.items():
        if tags_exclude & tags:
            continue  # skip if any excluded tag is present
        if tags_include_all and not tags_include_all <= tags:
            continue  # must contain all required tags
        if tags_include_any and not tags & tags_include_any:
            continue  # must contain at least one from the union
        selected.append(feature)

    return selected


def get_target_feature(
        feature_tags: dict
) -> str:
    """
    From the feature_tags dictionary, extract the target feature name with tag 'target'. Must be unique.
    :param feature_tags: Dictionary mapping feature names to tags.
    :return: The target name (str)
    """
    target_features = get_features_by_tags(feature_tags, tags_include_any={"target"})
    if len(target_features) != 1:
        raise ValueError(
            "Expected exactly one target feature, got {}".format(len(target_features))
        )
    return target_features[0]
