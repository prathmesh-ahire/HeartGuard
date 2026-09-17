'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import { LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { LIFT_SPRING } from '@/lib/motion';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Overlay';
import { ErrorState } from '@/components/ui/States';
import { HistoryEmpty, HistoryLoading, NoMatches, ServiceUnavailable } from '@/components/ui/PageStates';
import { CompareView } from '@/components/predict/CompareView';
import { RecordDrawer } from '@/components/history/RecordDrawer';
import { TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import {
  ApiError,
  deleteAllHistory,
  deleteHistoryRecord,
  getHistoryRecord,
  listHistory,
  restoreHistoryRecord,
  type HistoryPage,
  type HistoryRecord,
} from '@/lib/api';
import {
  NO_FILTERS,
  compareItemFromRecord,
  filtersFromSearch,
  formatWhen,
  hasFilters,
  listParams,
  searchFromFilters,
  type HistoryFilters,
} from '@/lib/history';

/**
 * The History page (T132.1-T132.6).
 *
 * ## The server filters, the page asks
 *
 * Search, filters and paging are query parameters of `GET /api/history`, so the
 * list is always what TinyDB holds, never a browser-side subset. The filters are
 * also kept in the address, so a reload -- or a link from Analyse, which names a
 * `record` or a `batch` -- opens the same view.
 *
 * ## Delete has an undo, delete-all has a confirmation
 *
 * Deleting one entry removes it from the file at once; the undo asks the API to
 * put the same entry back (same id and date). Deleting everything is asked for
 * in a dialog first and has no undo.
 *
 * ## No number is made here
 *
 * Counts, confidences and page numbers are the API's display strings. The date
 * is the one thing formatted in the browser: a moment shown in the viewer's own
 * time zone, which the server cannot know.
 */

type Load =
  | { state: 'loading' }
  | { state: 'ready'; page: HistoryPage }
  | { state: 'failed'; error: ApiError | Error };

const TASKS = prediction.tasks;

function classesOf(task: string): readonly string[] {
  return TASKS.find((entry) => entry.task === task)?.classes ?? [];
}

const INPUT = 'rounded-lg border border-line bg-panel px-3 py-1.5 text-body-md text-ink';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

export function HistoryBrowser() {
  const reduced = useReducedMotion();
  const [filters, setFilters] = useState<HistoryFilters | null>(null);
  const baseId = useId();
  const fieldId = (name: string) => baseId + name;
  const [search, setSearch] = useState('');
  const [load, setLoad] = useState<Load>({ state: 'loading' });
  const [reload, setReload] = useState(0);
  const [open, setOpen] = useState<HistoryRecord | null>(null);
  const [linkProblem, setLinkProblem] = useState<string | null>(null);
  const [selected, setSelected] = useState<HistoryRecord[]>([]);
  const [comparing, setComparing] = useState(false);
  const [undo, setUndo] = useState<HistoryRecord | null>(null);
  const [actionProblem, setActionProblem] = useState<string | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);
  const [emptied, setEmptied] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The address is read once, after hydration: the static HTML has no query.
  useEffect(() => {
    const initial = filtersFromSearch(window.location.search);
    setFilters(initial);
    setSearch(initial.q);
    const record = new URLSearchParams(window.location.search).get('record');
    if (record !== null && record !== '') {
      getHistoryRecord(record)
        .then(setOpen)
        .catch((error: unknown) => {
          setLinkProblem(
            error instanceof ApiError && error.status === 404
              ? 'The analysis this link points to is no longer in History.'
              : error instanceof Error
                ? error.message
                : String(error),
          );
        });
    }
  }, []);

  useEffect(() => {
    if (filters === null) return undefined;
    window.history.replaceState(null, '', window.location.pathname + searchFromFilters(filters, open?.id ?? null));
    return undefined;
  }, [filters, open]);

  useEffect(() => {
    if (filters === null) return undefined;
    let current = true;
    setLoad((previous) => (previous.state === 'ready' ? previous : { state: 'loading' }));
    listHistory(listParams(filters))
      .then((page) => {
        if (!current) return;
        if (page.total > 0 && filters.page > page.pages) {
          setFilters({ ...filters, page: page.pages });
          return;
        }
        setLoad({ state: 'ready', page });
      })
      .catch((error: unknown) => {
        if (current) setLoad({ state: 'failed', error: error instanceof Error ? error : new Error(String(error)) });
      });
    return () => {
      current = false;
    };
  }, [filters, reload]);

  const change = useCallback((patch: Partial<HistoryFilters>) => {
    setFilters((current) => ({ ...(current ?? NO_FILTERS), ...patch, page: patch.page ?? 1 }));
  }, []);

  const onSearch = (text: string) => {
    setSearch(text);
    if (searchTimer.current !== null) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => change({ q: text.trim() }), 300);
  };

  // A search still waiting to apply would otherwise land after the clear.
  const cancelSearch = () => {
    if (searchTimer.current !== null) clearTimeout(searchTimer.current);
    searchTimer.current = null;
  };

  useEffect(() => cancelSearch, []);

  const clear = () => {
    cancelSearch();
    setSearch('');
    setFilters({ ...NO_FILTERS });
  };

  const refresh = () => setReload((value) => value + 1);

  const replaceRecord = (record: HistoryRecord) => {
    setOpen(record);
    setSelected((current) => current.map((item) => (item.id === record.id ? record : item)));
    refresh();
  };

  const remove = async (record: HistoryRecord) => {
    setActionProblem(null);
    try {
      await deleteHistoryRecord(record.id);
      setOpen(null);
      setSelected((current) => current.filter((item) => item.id !== record.id));
      setComparing(false);
      setUndo(record);
      setEmptied(null);
      refresh();
    } catch (error) {
      setActionProblem('The analysis was not deleted. ' + (error instanceof Error ? error.message : String(error)));
    }
  };

  const restore = async () => {
    if (undo === null) return;
    setActionProblem(null);
    try {
      await restoreHistoryRecord(undo.id);
      setUndo(null);
      refresh();
    } catch (error) {
      setUndo(null);
      setActionProblem('The deletion was not undone. ' + (error instanceof Error ? error.message : String(error)));
    }
  };

  const removeAll = async () => {
    setBusy(true);
    setActionProblem(null);
    try {
      const done = await deleteAllHistory();
      setConfirmAll(false);
      setOpen(null);
      setSelected([]);
      setComparing(false);
      setUndo(null);
      setEmptied(done.deleted_display);
      clear();
      refresh();
    } catch (error) {
      setActionProblem('History was not emptied. ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setBusy(false);
    }
  };

  const toggle = (record: HistoryRecord) => {
    setComparing(false);
    setSelected((current) =>
      current.some((item) => item.id === record.id)
        ? current.filter((item) => item.id !== record.id)
        : [...current, record].slice(-2),
    );
  };

  const resultOptions = useMemo(() => {
    const names = filters?.task ? classesOf(filters.task) : TASKS.flatMap((entry) => entry.classes);
    return Array.from(new Set(names));
  }, [filters?.task]);

  const page = load.state === 'ready' ? load.page : null;
  const filtered = filters !== null && hasFilters(filters);
  const pair = selected.length === 2 ? (selected as [HistoryRecord, HistoryRecord]) : null;

  return (
    <div className="space-y-6">
      {linkProblem !== null ? (
        <p role="alert" className={cn(TYPE_SCALE.caption, 'rounded border border-warn-line bg-warn-soft p-3 text-warn')}>
          {linkProblem}
        </p>
      ) : null}

      <section aria-label="Search and filters" className="rounded-xl border border-line bg-sunken p-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <div className="block sm:col-span-2 lg:col-span-3">
            <label htmlFor={fieldId('q')} className="label-micro block">Search file names, notes and tags</label>
            <input
              id={fieldId('q')}
              type="search"
              value={search}
              onChange={(event) => onSearch(event.target.value)}
              className={cn(INPUT, 'mt-1 w-full')}
            />
          </div>
          <div className="block">
            <label htmlFor={fieldId('task')} className="label-micro block">Check</label>
            <select
              id={fieldId('task')}
              value={filters?.task ?? ''}
              onChange={(event) => change({ task: event.target.value, result: '' })}
              className={cn(INPUT, 'mt-1 w-full')}
            >
              <option value="">All checks</option>
              {TASKS.map((entry) => (
                <option key={entry.task} value={entry.task}>
                  {entry.title}
                </option>
              ))}
            </select>
          </div>
          <div className="block">
            <label htmlFor={fieldId('result')} className="label-micro block">Result</label>
            <select
              id={fieldId('result')}
              value={filters?.result ?? ''}
              onChange={(event) => change({ result: event.target.value })}
              className={cn(INPUT, 'mt-1 w-full')}
            >
              <option value="">All results</option>
              {resultOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="block">
            <label htmlFor={fieldId('tag')} className="label-micro block">Tag</label>
            <input
              id={fieldId('tag')}
              type="text"
              value={filters?.tag ?? ''}
              onChange={(event) => change({ tag: event.target.value })}
              className={cn(INPUT, 'mt-1 w-full')}
            />
          </div>
          <div className="block">
            <label htmlFor={fieldId('from')} className="label-micro block">From</label>
            <input
              id={fieldId('from')}
              type="date"
              value={filters?.from ?? ''}
              max={filters?.to || undefined}
              onChange={(event) => change({ from: event.target.value })}
              className={cn(INPUT, 'mt-1 w-full')}
            />
          </div>
          <div className="block">
            <label htmlFor={fieldId('to')} className="label-micro block">To</label>
            <input
              id={fieldId('to')}
              type="date"
              value={filters?.to ?? ''}
              min={filters?.from || undefined}
              onChange={(event) => change({ to: event.target.value })}
              className={cn(INPUT, 'mt-1 w-full')}
            />
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {filters?.batch ? (
            <Badge tone="info">
              One batch only
            </Badge>
          ) : null}
          {filtered ? (
            <Button tone="ghost" size="sm" icon="close" onClick={clear}>
              Clear filters
            </Button>
          ) : null}
        </div>
      </section>

      {undo !== null ? (
        <div role="status" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-panel p-3">
          <span className={TYPE_SCALE.body}>
            Deleted <span className="font-mono">{undo.file_name}</span>.
          </span>
          <Button tone="secondary" size="sm" onClick={() => void restore()}>
            Undo
          </Button>
        </div>
      ) : null}
      {emptied !== null ? (
        <p role="status" className={cn(TYPE_SCALE.caption, 'text-ink-2')}>
          History was emptied ({emptied} deleted).
        </p>
      ) : null}
      {actionProblem !== null ? (
        <p role="alert" className={cn(TYPE_SCALE.caption, 'rounded border border-danger-line bg-danger-soft p-3 text-danger')}>
          {actionProblem}
        </p>
      ) : null}
      {page?.notice ? (
        <p role="note" className={cn(TYPE_SCALE.caption, 'rounded border border-warn-line bg-warn-soft p-3 text-warn')}>
          {page.notice}
        </p>
      ) : null}

      {load.state === 'loading' ? <HistoryLoading /> : null}
      {load.state === 'failed' ? (
        load.error instanceof ApiError && load.error.offline ? (
          <ServiceUnavailable onRetry={refresh} />
        ) : (
          <ErrorState title="History could not be loaded" detail={load.error.message} onRetry={refresh} />
        )
      ) : null}

      {page !== null && page.total === 0 ? filtered ? <NoMatches onClear={clear} /> : <HistoryEmpty /> : null}

      {page !== null && page.total > 0 ? (
        <section aria-label="Saved analyses" className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className={cn(TYPE_SCALE.body, 'text-ink-2')} data-testid="history-total">
              <span className="stat font-mono text-ink">{page.display.total}</span>{' '}
              {filtered ? 'matching analyses' : 'saved analyses'}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => setComparing(true)} disabled={pair === null}>
                Compare selected
              </Button>
              <Button tone="ghost" icon="close" onClick={() => setConfirmAll(true)}>
                Delete all
              </Button>
            </div>
          </div>

          <div className="overflow-x-auto rounded-lg border border-line">
            <LazyMotion features={loadFeatures} strict>
            <table className="min-w-full text-left text-sm">
              <thead className="border-b-2 border-line bg-sunken font-mono text-label-sm uppercase text-ink-3">
                <tr>
                  <th scope="col" className="px-3 py-2">
                    <span className="sr-only">Select to compare</span>
                  </th>
                  <th scope="col" className="px-3 py-2">File</th>
                  <th scope="col" className="px-3 py-2">Result</th>
                  <th scope="col" className="px-3 py-2">Check</th>
                  <th scope="col" className="px-3 py-2">Confidence</th>
                  <th scope="col" className="px-3 py-2">Saved</th>
                  <th scope="col" className="px-3 py-2">Tags</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((record) => (
                  <m.tr
                    key={record.id}
                    className="border-t border-line align-top"
                    data-record-id={record.id}
                    data-file={record.file_name}
                    whileHover={reduced ? undefined : { backgroundColor: 'rgb(var(--sunken))' }}
                    transition={LIFT_SPRING}
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        aria-label={'Select ' + record.file_name + ' to compare'}
                        checked={selected.some((item) => item.id === record.id)}
                        onChange={() => toggle(record)}
                        className="accent-accent"
                      />
                    </td>
                    <td className="max-w-[16rem] px-3 py-2">
                      <button
                        type="button"
                        onClick={() => setOpen(record)}
                        title={record.file_name}
                        aria-label={'Open ' + record.file_name}
                        className="block max-w-full truncate font-mono text-accent-strong underline-offset-2 hover:underline"
                      >
                        {record.file_name}
                      </button>
                    </td>
                    <td className="px-3 py-2" data-cell="result">
                      <Badge tone="info">{record.display.result}</Badge>
                      {record.low_confidence ? (
                        <Badge tone="warn" className="ml-1">
                          Low confidence
                        </Badge>
                      ) : null}
                    </td>
                    <td className="px-3 py-2 text-ink-2" data-cell="task">
                      {record.task_title}
                    </td>
                    <td className="stat px-3 py-2 font-mono" data-cell="confidence">
                      {record.display.confidence}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2 text-ink-2" data-cell="date">
                      {formatWhen(record.created_at)}
                    </td>
                    <td className="px-3 py-2" data-cell="tags">
                      <span className="flex flex-wrap gap-1">
                        {record.tags.map((tag) => (
                          <Badge key={tag} tone="neutral">
                            {tag}
                          </Badge>
                        ))}
                      </span>
                    </td>
                  </m.tr>
                ))}
              </tbody>
            </table>
            </LazyMotion>
          </div>

          <nav aria-label="Pages" className="flex flex-wrap items-center justify-between gap-3">
            <Button
              size="sm"
              onClick={() => filters !== null && change({ page: filters.page - 1 })}
              disabled={page.page <= 1}
            >
              Previous
            </Button>
            <span className={cn(TYPE_SCALE.caption, 'text-ink-2')} data-testid="history-page">
              Page {page.display.page} of {page.display.pages}
            </span>
            <Button
              size="sm"
              onClick={() => filters !== null && change({ page: filters.page + 1 })}
              disabled={page.page >= page.pages}
            >
              Next
            </Button>
          </nav>
          <p className={cn(TYPE_SCALE.caption, 'text-ink-3')}>
            Select two analyses to compare them side by side; the selection is kept across pages.
          </p>
        </section>
      ) : null}

      {comparing && pair !== null ? (
        <CompareView
          onClose={() => setComparing(false)}
          items={[compareItemFromRecord(pair[0], classesOf(pair[0].task)), compareItemFromRecord(pair[1], classesOf(pair[1].task))]}
        />
      ) : null}

      {open !== null ? (
        <RecordDrawer
          key={open.id}
          record={open}
          classes={classesOf(open.task)}
          onClose={() => setOpen(null)}
          onSaved={replaceRecord}
          onDelete={(record) => void remove(record)}
          onShowBatch={(batch) => {
            setOpen(null);
            cancelSearch();
            setSearch('');
            setFilters({ ...NO_FILTERS, batch });
          }}
        />
      ) : null}

      <Modal
        open={confirmAll}
        onClose={() => setConfirmAll(false)}
        title="Delete all history?"
        description="Every saved analysis, with its notes and tags, is removed from this machine. This cannot be undone."
        footer={
          <>
            <Button tone="ghost" onClick={() => setConfirmAll(false)}>
              Keep history
            </Button>
            <Button tone="primary" onClick={() => void removeAll()} disabled={busy}>
              {busy ? 'Deleting…' : 'Delete everything'}
            </Button>
          </>
        }
      >
        <p className={cn(TYPE_SCALE.body, 'text-ink-2')}>
          This deletes every saved analysis, including any the current search and filters hide.
        </p>
      </Modal>
    </div>
  );
}
