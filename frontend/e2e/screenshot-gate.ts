import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

/**
 * T119.4: the hard gate in front of every screenshot.
 *
 * Throws -- which aborts the whole Playwright run before a single page is
 * opened -- unless the displayed-value audit passed on exactly the site now in
 * `frontend/out/`. The check itself is Python's (`require_passed_audit`), so the
 * gate and the audit cannot disagree about what "audited" means.
 */
export default function screenshotGate(): void {
  const python =
    process.env.PV_PYTHON ??
    (process.platform === 'win32'
      ? resolve(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe')
      : resolve(__dirname, '..', '..', '.venv', 'bin', 'python'));
  const script = resolve(__dirname, '..', '..', 'scripts', '45_audit_displayed_values.py');
  const result = spawnSync(python, [script, '--gate'], { encoding: 'utf-8' });
  const said = (result.stdout ?? '') + (result.stderr ?? '');
  if (result.status !== 0) {
    throw new Error('No screenshot of an unaudited page (T119.4).\n' + said);
  }
  process.stdout.write(said);
}
