from tqdm import tqdm


def process_examples(examples):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    idxs = []
    image_paths = []
    temporal_image_paths = []
    temporal_predicates = []
    for index in progress:
        report_id = examples["id"][index]
        image_path = examples["image_path"][index]
        temporal_image_path = examples["temporal_image_path"][index]
        temporal_predicate = examples["temporal_predicate"][index]
        idxs.append(report_id)
        image_paths.append(image_path)

        temporal_image_paths.append(temporal_image_path)
        temporal_predicates.append(temporal_predicate)
    return (
        idxs,
        image_paths,
        temporal_image_paths,
        temporal_predicates,
    )
