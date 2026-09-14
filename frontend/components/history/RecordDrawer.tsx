'use client';

import { useId, useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Drawer } from '@/components/ui/Overlay';
import { EmptyState } from '@/components/ui/States';
import { WaveformPlayer } from '@/components/audio/WaveformPlayer';
import { TYPE_SCALE, seriesColor } from '@/lib/tokens';
import { updateHistoryRecord, type HistoryRecord } from '@/lib/api';
import { formatWhen, parseTags, recordAudio } from '@/lib/history';

/**
 * One History entry in full (T132.3): the waveform when the recording can still
 * be fetched, the whole result, and the two things that can be edited -- notes
 * and tags. A stored result is never editable (Phase 129).
 *
 * Every value is a display string from the API; numeric fields only size bars.
 */

const SOURCE_LABEL: Record<HistoryRecord['source_kind'], string> = {
  upload: 'Uploaded or recorded file',
  sample: 'Built-in sample recording',
  manual: 'Added by hand',
};

export function RecordDrawer({
  record,
  classes,
  onClose,
  onSaved,
  onDelete,
  onShowBatch,
}: {
  record: HistoryRecord;
  classes: readonly string[];
  onClose: () => void;
  onSaved: (record: HistoryRecord) => void;
  onDelete: (record: HistoryRecord) => void;
  onShowBatch: (batchId: string) => void;
}) {
  const [notes, setNotes] = useState(record.notes);
  const baseId = useId();
  const fieldId = (name: string) => baseId + name;
  const [tagText, setTagText] = useState(record.tags.join(', '));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const tags = parseTags(tagText);
  const changed = notes.trim() !== record.notes || JSON.stringify(tags) !== JSON.stringify(record.tags);
  const audio = recordAudio(record);
  const names = [
    ...classes.filter((name) => name in record.probabilities),
    ...Object.keys(record.probabilities).filter((name) => !classes.includes(name)),
  ];

  const save = async () => {
    setSaving(true);
    setSaved(false);
    setProblem(null);
    try {
      const updated = await updateHistoryRecord(record.id, { notes, tags });
      setNotes(updated.notes);
      setTagText(updated.tags.join(', '));
      setSaved(true);
      onSaved(updated);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Drawer
      open
      onClose={onClose}
      title={record.file_name}
      description={record.task_title + ' · ' + formatWhen(record.created_at)}
      footer={
        <Button tone="secondary" icon="close" onClick={() => onDelete(record)}>
          Delete this analysis
        </Button>
      }
    >
      <div className="space-y-6" data-record-id={record.id}>
        {audio !== null ? (
          <WaveformPlayer source={audio} label={record.file_name} />
        ) : (
          <EmptyState
            icon="waveform"
            title="Audio not kept"
            description="History keeps each result, never the recording, so there is no waveform to show."
          />
        )}

        <section aria-label="Result" className="rounded-xl border border-line bg-panel p-4">
          <p className="label-micro">{record.task_title}</p>
          <p className={cn(TYPE_SCALE.h2, 'mt-1 capitalize text-ink')} data-testid="record-result">
            {record.display.result}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Badge tone="neutral">
              Confidence <span data-testid="record-confidence">{record.display.confidence}</span>
            </Badge>
            {record.low_confidence ? <Badge tone="warn">Low confidence</Badge> : null}
          </div>
          <p className="label-micro mt-4">Probability of each category</p>
          <ul className="mt-2 space-y-2">
            {names.map((name, index) => {
              const value = record.probabilities[name];
              const top = name === record.result;
              return (
                <li key={name} data-probability={name}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className={cn('font-mono text-label-md uppercase', top ? 'text-accent-deep' : 'text-ink-2')}>
                      {name}
                    </span>
                    <span className="stat font-mono text-telemetry-sm text-ink">
                      {record.display.probabilities[name] ?? 'n/a'}
                    </span>
                  </div>
                  <div className="mt-1 h-2 w-full overflow-hidden rounded-full border border-line bg-sunken">
                    <div
                      className={cn('h-full rounded-full', !top && 'opacity-60')}
                      style={{ width: (typeof value === 'number' ? value * 100 : 0) + '%', backgroundColor: seriesColor(index) }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          <dl className={cn(TYPE_SCALE.caption, 'mt-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-ink-2')}>
            <dt className="text-ink-3">Saved</dt>
            <dd>{formatWhen(record.created_at)}</dd>
            <dt className="text-ink-3">Source</dt>
            <dd>{SOURCE_LABEL[record.source_kind] ?? record.source_kind}</dd>
            {record.batch_id !== null ? (
              <>
                <dt className="text-ink-3">Batch</dt>
                <dd>
                  <button
                    type="button"
                    className="text-accent-strong underline underline-offset-2"
                    onClick={() => onShowBatch(record.batch_id as string)}
                  >
                    Show the rest of this batch
                  </button>
                </dd>
              </>
            ) : null}
          </dl>
        </section>

        <section aria-label="Notes and tags" className="space-y-3">
          <div className="block">
            <label htmlFor={fieldId('notes')} className="label-micro block">Notes</label>
            <textarea
              id={fieldId('notes')}
              value={notes}
              maxLength={2000}
              rows={4}
              onChange={(event) => {
                setNotes(event.target.value);
                setSaved(false);
              }}
              className="mt-1 w-full rounded-lg border border-line bg-panel px-3 py-2 text-body-md text-ink"
            />
          </div>
          <div className="block">
            <label htmlFor={fieldId('tags')} className="label-micro block">Tags</label>
            <input
              id={fieldId('tags')}
              type="text"
              value={tagText}
              onChange={(event) => {
                setTagText(event.target.value);
                setSaved(false);
              }}
              placeholder="follow-up, clinic A"
              className="mt-1 w-full rounded-lg border border-line bg-panel px-3 py-1.5 text-body-md text-ink"
            />
          </div>
          <p className={cn(TYPE_SCALE.caption, '-mt-2 text-ink-3')}>Separate tags with commas.</p>
          {tags.length > 0 ? (
            <ul className="flex flex-wrap gap-1.5" aria-label="Tags to save">
              {tags.map((tag) => (
                <li key={tag}>
                  <Badge tone="info">{tag}</Badge>
                </li>
              ))}
            </ul>
          ) : null}
          <div className="flex flex-wrap items-center gap-3">
            <Button tone="primary" icon="check" onClick={() => void save()} disabled={!changed || saving}>
              {saving ? 'Saving…' : 'Save notes and tags'}
            </Button>
            {saved && !changed ? (
              <span role="status" className={cn(TYPE_SCALE.caption, 'text-good')}>
                Saved
              </span>
            ) : null}
          </div>
          {problem !== null ? (
            <p role="alert" className={cn(TYPE_SCALE.caption, 'rounded border border-danger-line bg-danger-soft p-2 text-danger')}>
              The notes and tags were not saved. {problem}
            </p>
          ) : null}
        </section>
      </div>
    </Drawer>
  );
}
