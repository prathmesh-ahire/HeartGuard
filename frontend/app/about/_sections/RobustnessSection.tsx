import Link from 'next/link';

import { FigurePanel } from '@/components/charts/FigurePanel';
import { FacetTable } from '@/components/table/FacetTable';
import { Disclosure } from '@/components/ui/Disclosure';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { table } from '@/lib/generated/tables';

/**
 * Robustness Analytics (T117.1; moved under About the Model by T127.3).
 *
 * A **server** component. Each block pairs a committed table -- filterable by
 * the columns that define its rows, through `FacetTable` -- with the figure
 * drawn from the same run. Filtering selects exported rows; it never recomputes
 * one.
 *
 * Sections rather than tabs, deliberately: Radix renders only the active tab, so
 * a tabbed layout would ship four of its five tables absent from the static HTML
 * and the displayed-value audit would never see them.
 */

interface Block {
  id: string;
  eyebrow: string;
  title: string;
  tableId: string;
  facets: readonly string[];
  figureId: string | null;
  body: string;
}

const BLOCKS: readonly Block[] = [
  {
    id: 'noise',
    eyebrow: 'Noise · T20',
    title: 'Recording quality and added noise',
    tableId: 'T20',
    facets: ['analysis', 'run', 'model_id', 'group'],
    figureId: 'G30',
    body:
      'Two designs share this table. The observational rows split the real recordings by the PP-08 quality flag; the interventional rows add white noise at fixed signal-to-noise ratios to the same held-out records (EXP-E1). Rows marked as not reported carry too few recordings to support a rate and are kept visible rather than dropped.',
  },
  {
    id: 'duration',
    eyebrow: 'Duration · T21',
    title: 'Recording length',
    tableId: 'T21',
    facets: ['analysis', 'run', 'model_id', 'group'],
    figureId: 'G31',
    body:
      'Observational rows band the recordings by their own length; interventional rows truncate long recordings to fixed windows (EXP-E2), so length is changed while the recording is not. The two answer different questions and are never pooled.',
  },
  {
    id: 'dataset',
    eyebrow: 'Dataset · T16',
    title: 'A different corpus: adult to paediatric',
    tableId: 'T16',
    facets: ['metric', 'level', 'rule'],
    figureId: 'G29',
    body:
      'The deployed binary model, fitted on PhysioNet 2016, applied unchanged to CirCor 2022 against its clinical Outcome label (EXP-D1). This is cross-dataset transfer from an adult cohort to a predominantly paediatric cohort. A large drop is the expected consequence of that population mismatch, not evidence that the method fails; the limitations tab gives the numbers behind that framing.',
  },
  {
    id: 'source',
    eyebrow: 'Recording source · T-S5',
    title: 'A held-out PhysioNet sub-collection',
    tableId: 'T-S5',
    facets: ['model_id'],
    figureId: null,
    body:
      'Leave-one-sub-collection-out cross-validation inside PhysioNet 2016 (EXP-F3): each fold tests on a recording setup the model never saw. This is the result that bounds every pooled figure on this site. AUC is rank-based, so a fall here is not a threshold problem; the recording source is confounded with the label and no model fitted to this corpus can separate the two.',
  },
  {
    id: 'location',
    eyebrow: 'Location · T22',
    title: 'Auscultation location',
    tableId: 'T22',
    facets: ['run', 'model_id', 'location'],
    figureId: 'G32',
    body:
      'CirCor records each patient at up to four standard positions plus rare extras. Each row scores one position on its own. A position with too few recordings for a rate is shown with its exclusion reason rather than with a number that one record could swing.',
  },
  {
    id: 'confidence',
    eyebrow: 'Confidence · T23',
    title: 'Calibration and confidence',
    tableId: 'T23',
    facets: ['run', 'model_id'],
    figureId: 'G34',
    body:
      'Whether a stated probability means what it says. In-domain and out-of-domain confidence behave differently, which is why confidence is reported per run and never pooled.',
  },
  {
    id: 'failures',
    eyebrow: 'Failures · T27',
    title: 'False positives and false negatives',
    tableId: 'T27',
    facets: [],
    figureId: 'G35',
    body:
      'Where the deployed model is wrong, broken down by the recording properties that the audit records. A failure pattern is an observation about this corpus, not a clinical finding.',
  },
];

export function RobustnessSection() {
  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Robustness Analytics"
        lede="How far the headline results hold when the conditions change: noisier or shorter recordings, a different auscultation position, an unseen recording setup, and a different population altogether. Every table can be narrowed by the columns that define its rows; the rows themselves are the committed ones."
        note={
          <span>
            Read these against the{' '}
            <Link
              href="/about/limitations/"
              className="text-accent-strong underline decoration-dotted underline-offset-2"
            >
              limitations
            </Link>
            . Every pooled PhysioNet figure elsewhere on this site is a within-corpus
            cross-validation and must be quoted as such.
          </span>
        }
      />

      <nav aria-label="Sections" className="flex flex-wrap gap-1.5">
        {BLOCKS.map((block) => (
          <a
            key={block.id}
            href={'#' + block.id}
            className="rounded-lg border border-line bg-panel px-2.5 py-1 font-mono text-label-md uppercase text-ink-2 transition-colors hover:border-accent-line hover:text-accent-strong"
          >
            {block.title}
          </a>
        ))}
      </nav>

      {BLOCKS.map((block, index) => {
        const source = table(block.tableId);
        return (
          <Disclosure
            key={block.id}
            id={block.id}
            defaultOpen={index === 0}
            summary={
              <span>
                <span className="mr-2 font-mono text-label-sm uppercase text-ink-3">
                  {block.eyebrow}
                </span>
                {source ? source.title : block.title}
              </span>
            }
          >
            <p className="mb-4 text-body-sm text-ink-2">{block.body}</p>
            <GlassCard bodyClassName="p-3">
              <FacetTable table={source} facets={block.facets} />
            </GlassCard>
            {block.figureId ? (
              <GlassCard className="mt-3" eyebrow={block.figureId}>
                <FigurePanel className="max-w-4xl" figureId={block.figureId} />
              </GlassCard>
            ) : null}
          </Disclosure>
        );
      })}
    </div>
  );
}
