# SPDX-License-Identifier: Apache-2.0
"""Independent checks for the pinned DropTest deck format."""

import hashlib
import math
import re
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


MATS = (1, 4, 5, 8, 9)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cards(path):
    result, key = {}, None
    for line in path.read_text().splitlines():
        text = line.strip()
        if text.startswith("/"):
            key = text
            require(key not in result, f"Duplicate card {key}")
            result[key] = []
        elif text and not text.startswith("#") and key:
            result[key].append(text)
    return result


def numbers(text):
    values = [float(x.replace("D", "E").replace("d", "e")) for x in text.split()]
    require(all(math.isfinite(value) for value in values), "Nonfinite deck value")
    return values


def physical_inputs(path):
    text = path.read_text()
    require(
        not re.search(r"^\s*/(?:INCLUDE|IMPORT)\b", text, re.MULTILINE | re.IGNORECASE),
        "External deck dependencies need explicit provenance support; refusing implicit includes",
    )
    data = cards(path)
    material = {}
    # Independently locate each material's E/nu header and following data line.
    current, waiting = None, False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/"):
            match = re.match(
                r"/MAT/(?:ELAST|PLAS_TAB)/(\d+)(?:/|$)", stripped, re.IGNORECASE
            )
            current = int(match.group(1)) if match else None
            waiting = False
        elif current in MATS and stripped.startswith("#"):
            waiting = bool(
                re.search(r"\bE\b", stripped)
                and re.search(r"\bnu\b", stripped, re.IGNORECASE)
            )
        elif current in MATS and waiting and stripped:
            require(current not in material, f"Repeated E entry for material {current}")
            material[current] = numbers(stripped.split("#")[0])[0]
            waiting = False
    require(set(material) == set(MATS), "Missing or unsupported material E cards")
    walls = {}
    for key, lines in data.items():
        if key.upper().startswith("/RWALL/PLANE/"):
            require(len(lines) >= 5, "Incomplete rigid wall")
            origin, endpoint = numbers(lines[3])[:3], numbers(lines[4])[:3]
            require(len(origin) == len(endpoint) == 3, "Invalid wall coordinates")
            vector = [b - a for a, b in zip(origin, endpoint)]
            require(sum(v * v for v in vector) > 0, "Zero wall normal")
            walls[key] = {"origin": origin, "vector": vector}
    require(bool(walls), "Missing planar wall")
    return material, walls


def engine_timing(path):
    data = cards(path)
    runs = [lines for key, lines in data.items() if key.startswith("/RUN/")]
    require(len(runs) == 1 and bool(runs[0]), "Expected one real /RUN final time")
    final = numbers(runs[0][0])[0]
    require(
        "/ANIM/DT" in data and bool(data["/ANIM/DT"]), "Real /ANIM/DT interval required"
    )
    values = numbers(data["/ANIM/DT"][0])
    require(len(values) == 2, "Unsupported /ANIM/DT schedule")
    start, interval = values
    require(final > start >= 0 and interval > 0, "Invalid engine time schedule")
    return {
        "final_time": final,
        "animation_start": start,
        "animation_interval": interval,
        "source": "parsed original engine /RUN and /ANIM/DT, unchanged",
    }


def verify_case(
    starter, engine, base_material, base_walls, params, generator, expected_engine_hash
):
    actual, walls = physical_inputs(starter)
    measurements = {}
    for mid in MATS:
        expected = base_material[mid] * float(
            params["e_scales"].get(str(mid), params["e_scales"].get(mid))
        )
        require(
            math.isclose(actual[mid], expected, rel_tol=6e-6, abs_tol=1e-12),
            f"Material {mid} E was not correctly changed: {actual[mid]} != {expected}",
        )
        measurements[str(mid)] = {"actual_E": actual[mid], "expected_E": expected}
    require(set(walls) == set(base_walls), "Rigid-wall identity changed")
    for key, wall in walls.items():
        base = base_walls[key]
        expected = generator._rotate_vector_xyz(
            base["vector"], params["rwall_orientation_deg"]
        )
        require(
            all(
                math.isclose(a, b, abs_tol=1.1e-6)
                for a, b in zip(wall["origin"], base["origin"])
            ),
            "Wall origin changed",
        )
        require(
            all(
                math.isclose(a, b, abs_tol=1.1e-6)
                for a, b in zip(wall["vector"], expected)
            ),
            "Wall rotation mismatch",
        )
        wall["expected_vector"] = list(expected)
    require(sha256(engine) == expected_engine_hash, "Copied engine changed")
    return {
        "materials": measurements,
        "walls": walls,
        "starter_sha256": sha256(starter),
        "engine_sha256": sha256(engine),
    }
