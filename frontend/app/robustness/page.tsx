import type { Metadata } from 'next';
import Link from 'next/link';

import { FigurePanel } from '@/components/charts/FigurePanel';
import { FacetTable } from '@/components/table/FacetTable';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Robustness Analytics',
  description:
    'How results move under noise, shortened recordings, a different corpus, a held-out recording source and a different auscultation location.',
};

/**
 * Robustness Analytics (T117.1).
 *
 * A **server** page. Each section pairs a committed table -- filterable by the
 * columns that define its rows, through `FacetTable` -- with the figure drawn
 * from the same run. Filtering selects exported rows; it never recomputes one.
 *
 * Sections rather than tabs, deliberately: Radix renders only the active tab, so
 * a tabbed page would ship four of its five tables absent from the static HTML
 * and the displayed-value audit would never see them.
 */

interface Section {
  id: string;
  eyebrow: string;
  title: string;
  tableId: string;
  facets: readonly string[];
  figureId: string | null;
  body: string;
}

const SECTIONS: readonly Section[] = [
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
      'The deployed binary model, fitted on PhysioNet 2016, applied unchanged to CirCor 2022 against its clinical Outcome label (EXP-D1). This is cross-dataset transfer from an adult cohort to a predominantly paediatric cohort. A large drop is the expected consequence of that population mismatch, not evidence that the method fails; the limitations page gives the numbers behind that framing.',
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

export default function Page() {
  return (
    <div className="space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Robustness Analytics</h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          How far the headline results hold when the conditions change: noisier or shorter
          recordings, a different auscultation position, an unseen recording setup, and a
          different population altogether. Every table can be narrowed by the columns that
          define its rows; the rows themselves are the committed ones.
        </p>
        <p className="mt-3 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
          Read these against the{' '}
          <Link href="/limitations/" className="underline">
            limitations
          </Link>
          . Every pooled PhysioNet figure elsewhere on this site is a within-corpus
          cross-validation and must be quoted as such.
        </p>
        <nav aria-label="Sections" className="mt-5 flex flex-wrap gap-2 text-sm">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={'#' + section.id}
              className="rounded border border-slate-300 px-2.5 py-1 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {section.title}
            </a>
          ))}
        </nav>
      </section>

      {SECTIONS.map((section) => {
        const source = table(section.tableId);
        return (
          <section key={section.id} id={section.id} className="scroll-mt-24">
            <SectionHeader
              eyebrow={section.eyebrow}
              title={source ? source.title : section.title}
              description={section.body}
              level={2}
            />
            <FacetTable className="mt-5" table={source} facets={section.facets} />
            {section.figureId ? (
              <FigurePanel className="mt-8 max-w-4xl" figureId={section.figureId} />
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
