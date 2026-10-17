"""Run software regression tests and save evidence, not model accuracy."""
import io, json, platform, sys, unittest
from datetime import datetime, timezone
from pathlib import Path
stream = io.StringIO()
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
Path('reports').mkdir(exist_ok=True)
Path('reports/software_tests.txt').write_text(stream.getvalue())
report = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), passed=result.wasSuccessful(), interpretation='Software regression tests, including scripted model doubles and mocked transports. Not clinical performance or live model validation.')
Path('reports/software_tests.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if result.wasSuccessful() else 1)
