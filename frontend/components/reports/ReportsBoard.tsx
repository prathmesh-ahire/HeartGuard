'use client';

import { useCallback, useEffect, useState } from 'react';
import { LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';
import { CARD_HOVER, PRESS_SPRING } from '@/lib/motion';
import { SURFACE, TYPE_SCALE, LAYOUT } from '@/lib/tokens';
import { formatWhen } from '@/lib/history';
import { PRINT_ROUTE } from '@/lib/routes';
import { prediction } from '@/lib/generated/prediction';
import { Button, ButtonLink } from '@/components/ui/Button';
import { GlassCard } from '@/components/ui/GlassCard';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { EmptyState, ErrorState } from '@/components/ui/States';
import {
  ApiError,
  exportHistoryCsv,
  listHistory,
  listRecentReports,
  modelSummaryReportPdf,
  recordingReportPdf,
  redownloadReport,
  saveDocument,
  type HistoryRecord,
  type RecentReport,
} from '@/lib/api';

const TASKS = prediction.tasks;

const INPUT_CLS = 'rounded-lg border border-line bg-panel px-3 py-1.5 text-body-sm text-ink';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

/** One action's status: idle, working, or failed with a message. */
type ActionState = { key: string; status: 'working' } | { key: string; status: 'failed'; message: string } | null;

function reportMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : error instanceof Error ? error.message : String(error);
}

// ---------------------------------------------------------------------------
// Per-recording report (T134.1 / T134.3)
// ---------------------------------------------------------------------------

function RecentRecordings({ onGenerated }: { onGenerated: () => void }) {
  const reduced = useReducedMotion();
  const [rows, setRows] = useState<HistoryRecord[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [action, setAction] = useState<ActionState>(null);

  useEffect(() => {
    listHistory({ page_size: '8', sort: 'created_at', order: 'desc' })
      .then((page) => setRows(page.items))
      .catch(() => setFailed(true));
  }, []);

  const download = useCallback(
    async (record: HistoryRecord) => {
      setAction({ key: record.id, status: 'working' });
      try {
        saveDocument(await recordingReportPdf(record.id));
        setAction(null);
        onGenerated();
      } catch (error) {
        setAction({ key: record.id, status: 'failed', message: reportMessage(error) });
      }
    },
    [onGenerated],
  );

  return (
    <m.div whileHover={reduced ? undefined : CARD_HOVER} transition={PRESS_SPRING}>
    <GlassCard as="section" ariaLabel="Per-recording report" eyebrow="One analysis" title="Per-recording report" className="rounded-3xl">
      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-3')}>
        A PDF of one analysis: result, probabilities, notes, date, and the waveform when the
        recording is still available (built-in samples only -- History never keeps audio). The
        print view opens the same record for the browser&apos;s own Print &rarr; Save as PDF.
      </p>
      {failed ? (
        <ErrorState title="The recent analyses could not be loaded" detail="" />
      ) : rows === null ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Loading...</p>
      ) : rows.length === 0 ? (
        <EmptyState
          icon="reports"
          title="No analyses yet"
          description="Analyse a recording first; its report can be generated from History afterwards."
          action={
            <ButtonLink href="/" tone="primary" icon="pulse">
              Analyse a recording
            </ButtonLink>
          }
        />
      ) : (
        <ul className="divide-y divide-line">
          {rows.map((record) => (
            <li key={record.id} className="flex flex-wrap items-center justify-between gap-3 py-3">
              <div className="min-w-0">
                <p className={cn(TYPE_SCALE.body, 'truncate font-medium text-ink')}>{record.file_name}</p>
                <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
                  {record.task_title} &middot; {record.display.result} &middot; {formatWhen(record.created_at)}
                </p>
                {action?.key === record.id && action.status === 'failed' ? (
                  <ErrorState
                    className="mt-2"
                    title="The report was not produced"
                    detail={action.message}
                  />
                ) : null}
              </div>
              <div className="flex shrink-0 gap-2">
                <ButtonLink href={PRINT_ROUTE + '?record=' + record.id} size="sm" icon="reports">
                  Print view
                </ButtonLink>
                <Button
                  size="sm"
                  tone="primary"
                  icon="reports"
                  disabled={action?.key === record.id && action.status === 'working'}
                  onClick={() => void download(record)}
                >
                  {action?.key === record.id && action.status === 'working' ? 'Preparing...' : 'Download PDF'}
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </GlassCard>
    </m.div>
  );
}

// ---------------------------------------------------------------------------
// Bulk export (T134.2)
// ---------------------------------------------------------------------------

function BulkExport({ onGenerated }: { onGenerated: () => void }) {
  const reduced = useReducedMotion();
  const [task, setTask] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [state, setState] = useState<'idle' | 'working' | 'failed'>('idle');
  const [message, setMessage] = useState('');

  const run = async () => {
    setState('working');
    setMessage('');
    try {
      saveDocument(await exportHistoryCsv({ task: task || undefined, dateFrom: dateFrom || undefined, dateTo: dateTo || undefined }));
      setState('idle');
      onGenerated();
    } catch (error) {
      setState('failed');
      setMessage(reportMessage(error));
    }
  };

  return (
    <m.div whileHover={reduced ? undefined : CARD_HOVER} transition={PRESS_SPRING}>
    <GlassCard as="section" ariaLabel="Bulk export" eyebrow="Many analyses" title="Bulk export" className="rounded-3xl">
      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-3')}>
        Every History row matching these filters, as one CSV -- the same values History shows, with
        no filter applied at all if left blank.
      </p>
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="label-micro">Check type</span>
          <select className={INPUT_CLS} value={task} onChange={(e) => setTask(e.target.value)}>
            <option value="">All checks</option>
            {TASKS.map((t) => (
              <option key={t.task} value={t.task}>
                {t.title}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="label-micro">From</span>
          <input type="date" className={INPUT_CLS} value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label-micro">To</span>
          <input type="date" className={INPUT_CLS} value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
        </label>
        <Button tone="primary" icon="reports" disabled={state === 'working'} onClick={() => void run()}>
          {state === 'working' ? 'Preparing...' : 'Export CSV'}
        </Button>
      </div>
      {state === 'failed' ? (
        <ErrorState className="mt-3" title="The report was not produced" detail={message} />
      ) : null}
    </GlassCard>
    </m.div>
  );
}

// ---------------------------------------------------------------------------
// Model summary (T134.5)
// ---------------------------------------------------------------------------

function ModelSummary({ onGenerated }: { onGenerated: () => void }) {
  const reduced = useReducedMotion();
  const [state, setState] = useState<'idle' | 'working' | 'failed'>('idle');
  const [message, setMessage] = useState('');

  const run = async () => {
    setState('working');
    setMessage('');
    try {
      saveDocument(await modelSummaryReportPdf());
      setState('idle');
      onGenerated();
    } catch (error) {
      setState('failed');
      setMessage(reportMessage(error));
    }
  };

  return (
    <m.div whileHover={reduced ? undefined : CARD_HOVER} transition={PRESS_SPRING}>
    <GlassCard as="section" ariaLabel="Model summary" eyebrow="Overall performance" title="Model summary" className="rounded-3xl">
      {SHOW_SCREENING_NOTICE ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-3')}>{prediction.disclaimer}</p>
      ) : null}
      <Button tone="primary" icon="reports" disabled={state === 'working'} onClick={() => void run()}>
        {state === 'working' ? 'Preparing...' : 'Download model summary PDF'}
      </Button>
      {state === 'failed' ? (
        <ErrorState className="mt-3" title="The report was not produced" detail={message} />
      ) : null}
    </GlassCard>
    </m.div>
  );
}

// ---------------------------------------------------------------------------
// Recent reports (T134.4)
// ---------------------------------------------------------------------------

function RecentReports({ refreshKey }: { refreshKey: number }) {
  const reduced = useReducedMotion();
  const [rows, setRows] = useState<RecentReport[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    setFailed(null);
    listRecentReports()
      .then((page) => setRows(page.items))
      .catch((error: unknown) => setFailed(reportMessage(error)));
  }, [refreshKey]);

  const download = async (row: RecentReport) => {
    setBusy(row.id);
    try {
      saveDocument(await redownloadReport(row.id));
    } finally {
      setBusy(null);
    }
  };

  return (
    <m.div whileHover={reduced ? undefined : CARD_HOVER} transition={PRESS_SPRING}>
    <GlassCard as="section" ariaLabel="Recently generated reports" eyebrow="Recent downloads" title="Recently generated reports" className="rounded-3xl">
      {failed !== null ? (
        <ErrorState title="The report list could not be loaded" detail={failed} />
      ) : rows === null ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Loading...</p>
      ) : rows.length === 0 ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
          Nothing generated yet in this session. Reports made above appear here for re-download.
        </p>
      ) : (
        <ul className="divide-y divide-line">
          {rows.map((row) => (
            <li key={row.id} className="flex flex-wrap items-center justify-between gap-3 py-2.5">
              <div className="min-w-0">
                <p className={cn(TYPE_SCALE.body, 'truncate font-medium text-ink')}>{row.title}</p>
                <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
                  {formatWhen(row.created_at)} &middot; {row.size_display} bytes
                </p>
              </div>
              <Button size="sm" icon="reports" disabled={busy === row.id} onClick={() => void download(row)}>
                {busy === row.id ? 'Preparing...' : 'Re-download'}
              </Button>
            </li>
          ))}
        </ul>
      )}
    </GlassCard>
    </m.div>
  );
}

// ---------------------------------------------------------------------------
// ReportsBoard -- main export
// ---------------------------------------------------------------------------

/** Phase 134 -- the Reports page (T134.1-T134.6). */
export function ReportsBoard() {
  const [refreshKey, setRefreshKey] = useState(0);
  const bump = useCallback(() => setRefreshKey((v) => v + 1), []);

  return (
    <div className={LAYOUT.section}>
      <SectionHeader
        eyebrow="Download"
        title="Reports"
        description="Every number below comes from a stored result or the same source as the rest of the dashboard, formatted once in Python."
      />
      <LazyMotion features={loadFeatures} strict>
      {/* One column, not two side by side: one analysis, then many analyses,
          then the model summary, then recent downloads -- in that order, one
          under the next, not split into competing columns. */}
      <div className="space-y-6">
        <RecentRecordings onGenerated={bump} />
        <BulkExport onGenerated={bump} />
        <ModelSummary onGenerated={bump} />
        <RecentReports refreshKey={refreshKey} />
      </div>
      </LazyMotion>
    </div>
  );
}
