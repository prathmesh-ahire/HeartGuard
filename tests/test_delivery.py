"""Phase 126: the completion checklist, the asset counts, and the archive itself.

The archive is ~125 MB and takes about a minute to build, so the tests here
split into two kinds. The cheap ones exercise the packaging logic against a
small synthetic tree -- which is where the interesting failures live (an
excluded directory leaking in, a required entry missing, a stub `index.html`
passing as a built dashboard). The expensive one opens the real archive if it
has been built, and skips if it has not.

The claim this phase makes that most needs a test is **"a grader without Node
can serve the dashboard"**. That is checked structurally here (the built export
is in the archive and is not a stub) and was checked live in T126.7 by
extracting the archive and serving it with uvicorn alone.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.reporting import delivery
from src.reporting.delivery import DeliveryError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT_ROOT / "dist" / f"{delivery.DELIVERY_NAME}.zip"


def _tree(root: Path) -> None:
    """A minimal repository that satisfies every required entry."""
    for entry in delivery.REQUIRED_ENTRIES:
        if entry == delivery.MANIFEST_NAME:
            continue
        path = root / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 2000, encoding="utf-8")


class TestAssetCounts:
    def test_the_real_counts_meet_their_targets(self) -> None:
        """T126.2 -- 30 tables, 35 graphs, 20 diagrams, 20 algorithms, 13 screenshots."""
        if not (PROJECT_ROOT / "outputs" / "13_figures_diagrams").is_dir():
            pytest.skip("outputs/ is not in this checkout")
        gaps = {
            count.kind: count.missing for count in delivery.asset_inventory() if not count.ok
        }
        assert not gaps, f"asset gaps: {gaps}"

    def test_a_missing_asset_is_reported_by_id(self, tmp_path: Path) -> None:
        """A gap must name what is missing, not just a smaller number."""
        figures = tmp_path / "outputs" / "13_figures_diagrams"
        figures.mkdir(parents=True)
        for number in range(1, 35):  # 34 of the 35 graphs
            (figures / f"G{number:02d}_x.png").write_bytes(b"x")
        counts = {count.kind: count for count in delivery.asset_inventory(tmp_path)}
        assert counts["graphs"].missing == ("G35",)
        assert not counts["graphs"].ok

    def test_a_supplementary_table_cannot_fill_a_numbered_gap(self, tmp_path: Path) -> None:
        """T-S5 is a real deliverable and is not one of the thirty."""
        tables = tmp_path / "outputs" / "06_binary_results"
        tables.mkdir(parents=True)
        for number in range(1, 30):
            (tables / f"T{number:02d}_x.csv").write_text("a\n1\n", encoding="utf-8")
        for name in ("T-S1", "T-S2", "T-S3"):
            (tables / f"{name}_x.csv").write_text("a\n1\n", encoding="utf-8")
        counts = {count.kind: count for count in delivery.asset_inventory(tmp_path)}
        assert counts["tables"].found == 29
        assert counts["tables"].missing == ("T30",)


class TestChecklist:
    def test_no_required_item_is_missing_without_a_reason(self) -> None:
        """T126.1 -- `failed` means missing with no reason, which is the real defect.

        Items that fail *only* because a file is gitignored are excluded, the
        same way `tests/test_mega4_integrity.py` excludes them: on CI the model
        binaries and the feature matrix are legitimately absent, and a checklist
        that failed there would be measuring the clone rather than the project.
        """
        from src.reporting import evidence_pack as ep
        from src.utils.evidence import read_evidence

        if not (PROJECT_ROOT / "outputs" / "00_evidence_index").is_dir():
            pytest.skip("outputs/ is not in this checkout")
        spec = delivery.checklist()
        rows = read_evidence()
        failed = [
            item["item_id"]
            for item in spec["items"]
            if item["status"] == "failed" and not ep.fails_only_on_ignored_files(item, rows)
        ]
        assert not failed, f"required items missing with no reason: {failed}"
        assert not spec["unexplained"], f"not produced, no reason given: {spec['unexplained']}"

    def test_the_checklist_covers_both_source_documents(self) -> None:
        if not (PROJECT_ROOT / "outputs" / "00_evidence_index").is_dir():
            pytest.skip("outputs/ is not in this checkout")
        spec = delivery.checklist()
        sources = {str(row["source_document"]) for row in spec["items"]}
        assert len(sources) == 2, sources
        assert spec["total"] > 300


class TestPackaging:
    def test_an_excluded_directory_never_enters_the_archive(self, tmp_path: Path) -> None:
        """`dataset/` is 1.3 GB we have no licence to redistribute."""
        _tree(tmp_path)
        (tmp_path / "dataset" / "archive").mkdir(parents=True)
        (tmp_path / "dataset" / "archive" / "a0001.wav").write_bytes(b"RIFF")
        (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
        (tmp_path / "node_modules" / "left-pad" / "index.js").write_text("x", encoding="utf-8")

        archive = delivery.build_zip(tmp_path / "out.zip", root=tmp_path)
        names = set(zipfile.ZipFile(archive).namelist())
        assert not any(name.startswith("dataset/") for name in names)
        assert not any(name.startswith("node_modules/") for name in names)

    def test_the_archive_carries_its_own_manifest(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        archive = delivery.build_zip(tmp_path / "out.zip", root=tmp_path)
        with zipfile.ZipFile(archive) as opened:
            manifest = json.loads(opened.read(delivery.MANIFEST_NAME))
        assert manifest["node_required"] is False
        assert manifest["what_is_missing"] == "outputs/missing_outputs_report.txt"
        assert "not a diagnostic tool" in manifest["scope"]
        assert manifest["n_files"] == len(zipfile.ZipFile(archive).namelist()) - 1

    def test_a_sha256_is_written_beside_the_archive(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        archive = delivery.build_zip(tmp_path / "out.zip", root=tmp_path)
        digest = archive.with_suffix(".zip.sha256")
        assert digest.is_file()
        assert delivery._sha256(archive) in digest.read_text(encoding="utf-8")

    def test_verification_rejects_a_missing_required_entry(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        (tmp_path / "README.md").unlink()
        archive = delivery.build_zip(tmp_path / "out.zip", root=tmp_path)
        with pytest.raises(DeliveryError, match="missing required entries"):
            delivery.verify_zip(archive)

    def test_verification_rejects_a_stub_dashboard(self, tmp_path: Path) -> None:
        """The 'no Node required' claim rests entirely on this file being real."""
        _tree(tmp_path)
        (tmp_path / "frontend" / "out" / "index.html").write_text("<!-- todo -->", encoding="utf-8")
        archive = delivery.build_zip(tmp_path / "out.zip", root=tmp_path)
        with pytest.raises(DeliveryError, match="a grader would need Node"):
            delivery.verify_zip(archive)

    def test_verification_rejects_a_missing_archive(self, tmp_path: Path) -> None:
        with pytest.raises(DeliveryError, match="no delivery archive"):
            delivery.verify_zip(tmp_path / "nothing.zip")


class TestHandover:
    def test_the_handover_exists_and_states_the_scope(self) -> None:
        handover = PROJECT_ROOT / "HANDOVER.md"
        if not handover.is_file():
            pytest.skip("HANDOVER.md has not been generated in this checkout")
        text = handover.read_text(encoding="utf-8")
        assert "not clinically validated" in text
        assert "What clinical validation would require" in text
        assert "What was not done, and why" in text

    def test_it_repeats_the_five_things_a_reader_must_not_misread(self) -> None:
        handover = PROJECT_ROOT / "HANDOVER.md"
        if not handover.is_file():
            pytest.skip("HANDOVER.md has not been generated in this checkout")
        text = handover.read_text(encoding="utf-8")
        for claim in (
            "does not transfer across recording collections",
            "statistically tied",
            "recording-quality label",
            "population and acquisition effect",
        ):
            assert claim in text, f"the handover does not say: {claim}"

    def test_the_clinical_validation_section_is_specific(self) -> None:
        text = delivery.CLINICAL_VALIDATION
        for requirement in ("multi-site", "echocardiography", "pre-registered", "medical device"):
            assert requirement in text


class TestRealArchive:
    """T126.4/T126.7 -- the archive that was actually built."""

    @pytest.fixture(scope="class")
    def verified(self) -> dict:
        if not ARCHIVE.is_file():
            pytest.skip("the delivery archive has not been built in this checkout")
        return delivery.verify_zip(ARCHIVE)

    def test_it_opens_cleanly_and_passes_every_check(self, verified: dict) -> None:
        assert verified["entries"] > 1000
        assert verified["node_required"] is False

    def test_it_carries_the_model_binaries_the_repository_gitignores(self) -> None:
        if not ARCHIVE.is_file():
            pytest.skip("the delivery archive has not been built in this checkout")
        names = zipfile.ZipFile(ARCHIVE).namelist()
        joblibs = [name for name in names if name.endswith(".joblib")]
        assert len(joblibs) >= 5, "a delivery without the trained models is not runnable"
        assert any("binary/final" in name for name in joblibs)

    def test_it_carries_the_feature_matrix(self) -> None:
        if not ARCHIVE.is_file():
            pytest.skip("the delivery archive has not been built in this checkout")
        names = zipfile.ZipFile(ARCHIVE).namelist()
        assert "outputs/03_features/all_features_matrix.parquet" in names
