import type { Metadata } from 'next';

import { DatasetExplorer } from '@/app/dataset/DatasetExplorer';
import { GroupedBars } from '@/components/charts/Charts';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { ResultsTable } from '@/components/table/ResultsTable';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { datasetSummary } from '@/lib/generated';
import { figure } from '@/lib/generated/figures';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Dataset Overview',
  description: 'The four public PCG corpora as audited against the files on disk.',
};

/**
 * Dataset Overview (T114.2, T114.3).
 *
 * Server component; the filterable record table is the one client island.
 *
 * ## The scope trap this page is built around
 *
 * T01 counts every file on disk. T02 counts labelled records per class. The
 * duration summary in T03 covers the supervised subset only. Those are three
 * different populations, and note.md records a table that once put 21.98 corpus
 * hours beside 3,240 supervised records on the same row — both numbers correct,
 * the row arithmetically impossible.
 *
 * So every block here states its own population in its heading or its caption,
 * the tiles show the corpus count and the modelled count side by side, and the
 * class-distribution chart is labelled as covering labelled records only. A
 * reader comparing the bars in G01 against the bins in G04 will find they do not
 * sum to the same total, and will find the reason written next to them.
 */
export default function Page() {
  return (
    <div className="space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Dataset Overview</h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          Four public phonocardiogram corpora, audited file by file rather than taken
          from their documentation. Where the published counts and the files on disk
          disagreed, the files won and the discrepancy is recorded.
        </p>
        <p className="mt-3 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
          {datasetSummary.scope_note}
        </p>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Per corpus"
          title="Files on disk, and the subset actually modelled"
          level={2}
        />
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {datasetSummary.summary.map((row) => (
            <StatTile
              key={row.dataset_source}
              label={row.dataset_name}
              display={row.n_modelled_display}
              value={row.n_modelled}
              unit="modelled"
              source={datasetSummary.source}
              hint={
                <>
                  {row.n_files_display} files · {row.n_subjects_display} subjects ·{' '}
                  {row.hours_modelled_display} h modelled
                </>
              }
            />
          ))}
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="T01" title="Dataset inventory" level={2} />
        <ResultsTable
          className="mt-5"
          table={table('T01')}
          caption="Every file found on disk, with the subject-identifier origin per corpus. Counts here are the whole folder, not the modelled subset."
        />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T02"
          title="Class distribution and imbalance"
          description="Labelled records only, per task. The five label spaces are separate and are never merged: a row belongs to exactly one of them."
          level={2}
        />
        <ResultsTable className="mt-5" table={table('T02')} />

        <div className="mt-8 grid gap-8 lg:grid-cols-2">
          <div>
            <GroupedBars
              source={figure('G02')}
              categoryColumn="class"
              valueColumns={['n_records']}
              label="Records per class, per dataset"
              caption="G02 — labelled records per class. Counts, not shares."
              height={340}
            />
            <FigureDownload figureId="G02" className="mt-2" />
          </div>
          <div>
            <GroupedBars
              source={figure('G03')}
              categoryColumn="class"
              valueColumns={['share']}
              label="Class share within each dataset"
              caption="G03 — the same records as a share of their own dataset, which is what makes the imbalance comparable across corpora of very different sizes."
              height={340}
            />
            <FigureDownload figureId="G03" className="mt-2" />
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T03"
          title="Recording duration and sampling"
          description="Duration statistics over the supervised subset. Corpus-wide hours are in T01 and are a different population; the two are deliberately not shown on the same row."
          level={2}
        />
        <ResultsTable className="mt-5" table={table('T03')} />
        <div className="mt-8">
          <GroupedBars
            source={figure('G04')}
            categoryColumn="bin_low_sec"
            valueColumns={['n_records']}
            label="Recording duration histogram"
            caption="G04 — duration histogram over supervised records. The bin totals here and the bars in G01 count different populations and do not sum to the same number."
            height={320}
          />
          <FigureDownload figureId="G04" className="mt-2" />
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T114.3"
          title="Every audited recording"
          description="Filter by corpus, subset, label or flag, and search by record or subject identifier. The whole audited corpus is loaded — nothing is sampled or truncated."
          level={2}
        />
        <div className="mt-6">
          <DatasetExplorer />
        </div>
      </section>
    </div>
  );
}
