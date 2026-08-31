import { objectives } from '@/lib/generated';

/**
 * The six locked objectives (T114.1).
 *
 * A **server** component with no state, rendering `generated/objectives.json`
 * exactly as Python emitted it. The wording is never edited here, never
 * truncated, never wrapped in a shortened summary — the blueprint's own
 * instruction is that these are not to be changed, shortened, paraphrased or
 * reworded anywhere, and a page that "tightened" one would be the most likely
 * place for that to happen.
 *
 * `handle` labels a card; `wording` is its body. The handle is never rendered
 * in place of the wording, only above it.
 *
 * Each objective is published with the sha256 of its own text, so T125.4's
 * verbatim check is a comparison rather than a reading exercise.
 */
export function Objectives({ className }: { className?: string }) {
  return (
    <div className={className}>
      <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-400">
        {objectives.locked_notice}
      </p>

      <ol className="mt-6 grid gap-4 lg:grid-cols-2">
        {objectives.objectives.map((objective) => (
          <li
            key={objective.number}
            className="rounded-lg border border-slate-200 bg-white/60 p-5 dark:border-slate-800 dark:bg-slate-900/40"
          >
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-sky-700 dark:text-sky-400">
                {objective.label}
              </h3>
              <span
                className={
                  objective.status === 'produced'
                    ? 'rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300'
                    : 'rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-300'
                }
              >
                {objective.status === 'produced' ? 'evidence produced' : 'evidence pending'}
              </span>
            </div>

            <p className="mt-1 text-sm font-medium text-slate-800 dark:text-slate-200">
              {objective.handle}
            </p>

            {/* The locked wording. Quoted exactly; nothing added inside it. */}
            <blockquote className="mt-3 border-l-2 border-slate-300 pl-4 text-slate-700 dark:border-slate-700 dark:text-slate-300">
              {objective.wording}
            </blockquote>

            {objective.caveat ? (
              <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">{objective.caveat}</p>
            ) : null}

            {objective.pending_reason ? (
              <p className="mt-3 text-sm text-amber-800 dark:text-amber-300">
                {objective.pending_reason}
              </p>
            ) : null}

            <dl className="mt-4 space-y-1 text-xs text-slate-500 dark:text-slate-500">
              <div className="flex gap-2">
                <dt className="shrink-0">Implemented in</dt>
                <dd className="font-mono">{objective.modules.join(', ')}</dd>
              </div>
              {objective.evidence.length > 0 ? (
                <div className="flex gap-2">
                  <dt className="shrink-0">Evidence</dt>
                  <dd className="font-mono">
                    {objective.evidence.map((item) => item.dir).join(', ')}
                  </dd>
                </div>
              ) : null}
              <div className="flex gap-2">
                <dt className="shrink-0">Wording sha256</dt>
                <dd className="font-mono">{objective.wording_sha256.slice(0, 16)}</dd>
              </div>
            </dl>
          </li>
        ))}
      </ol>

      <div className="mt-6 rounded border border-slate-200 p-4 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-500">
        <p>
          Quoted from {objectives.source}, section 1, page {objectives.source_page}. Source
          sha256 <span className="font-mono">{objectives.source_sha256.slice(0, 16)}</span>.
        </p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {objectives.transcription_notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
