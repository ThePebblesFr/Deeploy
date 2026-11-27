from pathlib import Path

"""
Substitutes roundrobin_template.c's MODEL1_TAG/MODEL2_TAG placeholders with the two
real model tags (e.g. TinyClassifier_heavy/AnomalyDetection_heavy).

Deliberately a plain literal-substring replace, NOT a \b-anchored regex like
update_dynmultideeploy_test_main()'s Model_1/MODEL_1 pattern: that pattern's
placeholder always STARTS a fresh identifier (Model_1_inputs, Model_1_output_0, ...),
so a \b right before it lands on a true boundary. MODEL1_TAG/MODEL2_TAG appear as a
SUFFIX glued onto another prefix instead (InitNetwork_MODEL1_TAG,
RunNetwork_MODEL1_TAG) -- there's no word boundary between the "_" and "M", since
both are \\w, so a \b-anchored version of this pattern would silently fail to match
at all. A plain substring replace has no such issue and is exact here since the
placeholder tokens are unique, uppercase, and chosen specifically not to collide
with anything else in the template.
"""

ROUNDROBIN_TEMPLATE_PATH = Path(
    "/app/Deeploy/DynMultiDeeployTest/Platforms/RoundRobin/src/roundrobin_template.c"
)
ROUNDROBIN_OUTPUT_PATH = Path(
    "/app/Deeploy/DynMultiDeeployTest/Platforms/RoundRobin/src/roundrobin.c"
)

PRIORITY_SEQUENTIAL_TEMPLATE_PATH = Path(
    "/app/Deeploy/DynMultiDeeployTest/Platforms/PrioritySequential/src/priority_sequential_template.c"
)
PRIORITY_SEQUENTIAL_OUTPUT_PATH = Path(
    "/app/Deeploy/DynMultiDeeployTest/Platforms/PrioritySequential/src/priority_sequential.c"
)


def _substitute(template_path: Path, output_path: Path, model1_tag: str, model2_tag: str, log_tag: str) -> None:
    content = template_path.read_text()
    content = content.replace("MODEL1_TAG", model1_tag).replace("MODEL2_TAG", model2_tag)
    output_path.write_text(content)
    print(f"[{log_tag}] Substituted MODEL1_TAG -> {model1_tag}, MODEL2_TAG -> {model2_tag} in {output_path}")


def update_roundrobin_main(model1_tag: str, model2_tag: str) -> None:
    _substitute(ROUNDROBIN_TEMPLATE_PATH, ROUNDROBIN_OUTPUT_PATH, model1_tag, model2_tag, "RoundRobin")


def update_priority_sequential_main(model1_tag: str, model2_tag: str) -> None:
    _substitute(PRIORITY_SEQUENTIAL_TEMPLATE_PATH, PRIORITY_SEQUENTIAL_OUTPUT_PATH, model1_tag, model2_tag,
                "PrioritySequential")
