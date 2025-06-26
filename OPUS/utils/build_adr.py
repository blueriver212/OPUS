import re

_FIELD_RE = re.compile(r"^(year|shell|species|num_remove)_(\d+)$", re.I)

def build_adr_schedule(econ_params):
    """
    Building up what ADR gets implemented, with the ability to remove from multiple shells, species, and across multiple years
    Look at attributes on econ_params named like
        year_1, shell_1, species_1, num_remove_1,
        year_2, shell_2, ...
    and assemble a list of dicts ready for apply_ADR().
    """
    buckets = {}                       # {idx: {field: value}}

    for name, value in vars(econ_params).items():
        m = _FIELD_RE.match(name)
        if not m:
            continue
        field, idx = m.group(1).lower(), int(m.group(2))
        buckets.setdefault(idx, {})[field] = value

    schedule = []
    required = {"year", "shell", "species", "num_remove"}
    for idx in sorted(buckets):
        data = buckets[idx]
        missing = required - data.keys()
        if missing:
            raise ValueError(f"ADR set {idx}: missing {', '.join(missing)}")

        # tidy up types ------------------------------------------------------
        year_val = data["year"]
        if isinstance(year_val, str) and year_val.isdigit():
            year_val = int(year_val)

        schedule.append({
            "year":       year_val,                     # int or "every"
            "shell":      int(data["shell"]),
            "species":    data["species"],
            "num_remove": int(data["num_remove"]),
        })

    return schedule
