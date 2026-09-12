"""Offline tests only: no model, dataset, inference or evaluation imports."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checker():
    path = ROOT / 'scripts/verify_submission_package.py'
    assert path.is_file(), 'Submission checker must exist'
    spec = importlib.util.spec_from_file_location('submission_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence():
    return json.loads((ROOT / 'results/exp020-submission-evidence.json').read_text())


def test_frozen_evidence_contract():
    assert checker().evidence_errors(evidence()) == []


def test_checker_rejects_wrong_parameter_count():
    data = copy.deepcopy(evidence())
    data['summary']['parameter_count'] += 1
    assert 'parameter count' in checker().evidence_errors(data)


def test_checker_rejects_validation_not_matching_arithmetic():
    data = copy.deepcopy(evidence())
    data['validation'][-1]['tokens'] += 1
    assert 'validation token arithmetic' in checker().evidence_errors(data)


def test_checker_rejects_wrong_metric_or_checkpoint():
    data = copy.deepcopy(evidence())
    data['sources']['checkpoint']['sha256'] = '0' * 64
    assert 'checkpoint identity' in checker().evidence_errors(data)


def test_changed_submission_links_resolve():
    assert checker().link_errors(ROOT) == []


def test_figures_are_present_and_bound_to_evidence():
    assert checker().figure_errors(ROOT) == []


def test_submission_tools_have_no_execution_entrypoints_for_benchmarks():
    import ast
    for name in ('verify_submission_package.py', 'render_submission_figures.py', 'export_submission_evidence.py'):
        source = (ROOT / 'scripts' / name).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                assert not any(x in target for x in ('simple_evaluate', 'load_dataset', 'load_from_disk', 'os.system'))
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess':
                    assert target == 'subprocess.check_output'
                    assert ast.literal_eval(node.args[0]) == ['git', 'rev-parse', 'HEAD']
