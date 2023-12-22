import re

from tqdm import tqdm

from tokenizer import Tokenizer


def clean_report_iu_xray(report):
    report_cleaner = (
        lambda t: t.replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )

    def sent_cleaner(t):
        return re.sub(
            "[.,?;*!%^&_+():-\[\]{}]",
            "",
            t.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .strip()
            .lower(),
        )

    tokens = [
        sent_cleaner(sent)
        for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    report = " . ".join(tokens) + " ."
    return report


def clean_report_mimic_cxr(report):
    report_cleaner = (
        lambda t: t.replace("\n", " ")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )

    def sent_cleaner(t):
        return re.sub(
            "[.,?;*!%^&_+():-\[\]{}]",
            "",
            t.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .strip()
            .lower(),
        )

    tokens = [
        sent_cleaner(sent)
        for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    report = " . ".join(tokens) + " ."
    return report


def load_exemplar(annotation, text_tokenizer: Tokenizer, max_tgt_length):
    id2exemplar = {}
    progress = tqdm(annotation, desc="Loading Exemplars")
    for sample in progress:
        id2exemplar[sample["id"]] = []
    return id2exemplar


def jaccard_distance(a, b):
    return len(a & b) / len(a | b)


def insert_plan(report, tokenizer, splan, gplan, observation):
    report = tokenizer.clean_report(report)
    sentences = [s for s in report.split(".") if len(s.strip()) > 0]
    if len(sentences) == 0:
        return []
    no_finding = observation[-1]
    observation = observation[:-1]
    positions = sorted(splan.keys(), key=lambda x: int(x))
    planed_observation = set()
    observation_category = {o.split(":")[0] for o in observation}
    for pos in positions:
        planed_obs = splan[pos]["observation"]
        clean_planed_obs = []
        for o in planed_obs:
            c = o.split(":")[0]
            if c not in observation_category:
                continue
            if o not in observation:
                for new_o in observation:
                    if c in new_o:
                        o = new_o
                        clean_planed_obs.append(o)
                        break
            else:
                clean_planed_obs.append(o)
        splan[pos]["observation"] = clean_planed_obs
        planed_observation.update(splan[pos]["observation"])
    left_observation = [o for o in observation if o not in planed_observation]

    tokens = []
    existed_plan = set()
    for pos in positions:
        sentence_plan = ["[{}]".format(p) for p in splan[pos]["observation"]]
        sentence_plan = [o for o in sentence_plan if o not in existed_plan]

        if len(sentence_plan) > 1 and gplan is not None:
            sentence_plan = sorted(
                sentence_plan, key=lambda x: gplan[x[1:-1].split(":")[0]]
            )
        tokens.extend(sentence_plan)
        tokens.extend(splan[pos]["sentence"].strip().split())

    if len(left_observation) > 1:
        left_observation = sorted(
            left_observation, key=lambda x: gplan[x.split(":")[0]]
        )
    left_observation = [no_finding] + left_observation
    left_tokens = ["[{}]".format(o) for o in left_observation]
    tokens = left_tokens + tokens
    ids = []
    for token in tokens:
        if token == " ":
            continue
        ids.append(tokenizer.get_id_by_token(token))
    ids = ids + [tokenizer.eos_token_id]
    return ids


def construct_obs_aware_token(sentences, entity2id):
    obs_aware_tokens = []
    for sentence_id in sentences:
        sentence = sentences[sentence_id]
        observations = sentence["observation"]
        sentence = sentence["sentence"]
        tokens = sentence.split()
        for token in tokens:
            if token == " ":
                continue
            obs_aware_token_ids = set()
            if len(observations) == 0:
                observations = ["NONE"]
            for obs in observations:
                obs_aware_token = obs + "-" + token
                if obs_aware_token in entity2id:
                    obs_aware_token_ids.add(obs_aware_token)
            obs_aware_tokens.append(obs_aware_token_ids)
    return obs_aware_tokens


def process_examples(
    examples,
    max_tgt_length,
    tokenizer,
):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    labels = []
    idxs = []
    image_paths = []
    temporal_image_paths = []
    temporal_reports = []
    temporal_entity_ids = []
    current_entity_ids = []
    temporal_predicates = []
    for index in progress:
        report_id = examples["id"][index]
        image_path = examples["image_path"][index]
        report = tokenizer.encode(examples["report"][index])
        label = report[1:]
        if len(label) > max_tgt_length:
            label = label[: max_tgt_length - 1] + label[-1:]
        temporal_image_path = examples["temporal_image_path"][index]
        temporal_report = examples["temporal_report"][index]
        if temporal_report is None:
            temporal_report = ""
        if len(temporal_report) == 0:
            temporal_report = []
        else:
            temporal_report = tokenizer.encode(temporal_report)

        if len(temporal_report) > max_tgt_length:
            temporal_report = (
                temporal_report[: max_tgt_length - 1] + temporal_report[-1:]
            )
        temporal_predicate = examples["temporal_predicate"][index]
        labels.append(label)
        idxs.append(report_id)
        image_paths.append(image_path)

        temporal_image_paths.append(temporal_image_path)
        temporal_reports.append(temporal_report)
        temporal_predicates.append(temporal_predicate)
        temporal_entity_ids.append(examples["temporal_entity"][index])
        current_entity_ids.append(examples["current_entity"][index])
    return (
        idxs,
        image_paths,
        temporal_image_paths,
        temporal_entity_ids,
        current_entity_ids,
        temporal_predicates,
        temporal_reports,
        labels,
    )
