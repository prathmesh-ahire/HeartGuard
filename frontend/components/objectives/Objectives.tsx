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
      <p className="max-w-3xl text-sm text-ink-2">
        {objectives.locked_notice}
      </p>

      <ol className="mt-6 grid gap-4 lg:grid-cols-2">
        {objectives.objectives.map((objective) => (
          <li
            key={objective.number}
            className="rounded-lg border border-line bg-panel/60 p-5"
          >
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-accent-strong">
                {objective.label}
              </h3>
              <span
                className={
                  objective.status === 'produced'
                    ? 'rounded-full bg-good-soft px-2 py-0.5 text-xs text-good'
                    : 'rounded-full bg-warn-soft px-2 py-0.5 text-xs text-warn'
                }
              >
                {objective.status === 'produced' ? 'evidence produced' : 'evidence pending'}
              </span>
            </div>

            <p className="mt-1 text-sm font-medium text-ink">
              {objective.handle}
            </p>

            {/* The locked wording. Quoted exactly; nothing added inside it. */}
            <blockquote className="mt-3 border-l-2 border-line pl-4 text-ink-2">
              {objective.wording}
            </blockquote>

            {objective.caveat ? (
              <p className="mt-3 text-sm text-ink-2">{objective.caveat}</p>
            ) : null}

            {objective.pending_reason ? (
              <p className="mt-3 text-sm text-warn">
                {objective.pending_reason}
              </p>
            ) : null}
          </li>
        ))}
      </ol>

      <div className="mt-6 rounded border border-line p-4 text-xs text-ink-3">
        <p>Quoted from the project blueprint, section 1, page {objectives.source_page}.</p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {objectives.transcription_notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
