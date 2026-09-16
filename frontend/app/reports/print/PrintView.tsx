'use client';

import { useEffect, useState } from 'react';

import { cn } from '@/lib/cn';
import { DISCLAIMER_TEXT } from '@/components/Disclaimer';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';
import { formatWhen } from '@/lib/history';
import { TYPE_SCALE, SURFACE } from '@/lib/tokens';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/States';
import { ApiError, getHistoryRecord, type HistoryRecord } from '@/lib/api';

/**
 * T134.3 -- a print-friendly view of one History record, opened from its
 * "Print view" link on the Reports page. `?record=<id>` rather than a dynamic
 * route: `output: 'export'` cannot pre-render one page per History entry, and
 * the History page already reads its open record the same way (T132.3).
 *
 * The disclaimer here is NOT inside AppShell's `print:hidden` wrapper -- a
 * printed report must carry it (research rule 7), unlike the nav chrome.
 */
export function PrintView() {
  const [recordId, setRecordId] = useState<string | null | undefined>(undefined);
  const [record, setRecord] = useState<HistoryRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRecordId(new URLSearchParams(window.location.search).get('record'));
  }, []);

  useEffect(() => {
    if (!recordId) return;
    getHistoryRecord(recordId)
      .then(setRecord)
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)));
  }, [recordId]);

  if (recordId === undefined) return null;
  if (recordId === null) {
    return (
      <p className={cn(TYPE_SCALE.body, SURFACE.muted)}>
        No record given. Open this page from a recording&apos;s &quot;Print view&quot; link on the
        Reports page.
      </p>
    );
  }
  if (error !== null) {
    return <ErrorState title="This record could not be loaded" detail={error} />;
  }
  if (record === null) {
    return <p className={cn(TYPE_SCALE.body, SURFACE.muted)}>Loading...</p>;
  }

  const classes = Object.keys(record.display.probabilities);

  return (
    <div className="max-w-2xl space-y-6 print:max-w-none">
      <div className="print:hidden">
        <Button tone="primary" icon="reports" onClick={() => window.print()}>
          Print or save as PDF
        </Button>
      </div>

      <h1 className={TYPE_SCALE.h1}>PV-MEPCG / PulseVision -- Recording report</h1>
      {SHOW_SCREENING_NOTICE ? (
        <p className={cn(TYPE_SCALE.caption, 'font-semibold text-ink')}>{DISCLAIMER_TEXT}</p>
      ) : null}

      <div>
        <p className={TYPE_SCALE.h3}>{record.file_name}</p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
          {record.task_title} &middot; {formatWhen(record.created_at)}
        </p>
      </div>

      <dl className="grid grid-cols-[160px_1fr] gap-y-1.5 text-body-sm">
        <dt className={SURFACE.muted}>Result</dt>
        <dd data-testid="print-result">{record.display.result}</dd>
        <dt className={SURFACE.muted}>Confidence</dt>
        <dd data-testid="print-confidence">{record.display.confidence}</dd>
        <dt className={SURFACE.muted}>Low confidence</dt>
        <dd>{record.low_confidence ? 'Yes' : 'No'}</dd>
      </dl>

      <div>
        <p className={TYPE_SCALE.h3}>Class probabilities</p>
        <table className="mt-2 w-full text-body-sm">
          <tbody>
            {classes.map((name) => (
              <tr key={name} className="border-b border-line">
                <td className="py-1 pr-4">{name}</td>
                <td className="py-1 font-mono">{record.display.probabilities[name]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <p className={TYPE_SCALE.h3}>Notes</p>
        <p className={cn(TYPE_SCALE.body, SURFACE.muted)}>{record.notes || 'No notes recorded.'}</p>
      </div>

      <div>
        <p className={TYPE_SCALE.h3}>Tags</p>
        <p className={cn(TYPE_SCALE.body, SURFACE.muted)}>{record.tags.join(', ') || 'No tags.'}</p>
      </div>
    </div>
  );
}
