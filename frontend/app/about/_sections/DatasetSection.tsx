import { DatasetExplorer } from '@/app/about/_sections/DatasetExplorer';
import { GroupedBars } from '@/components/charts/Charts';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { datasetSummary } from '@/lib/generated';
import { G02 } from '@/lib/generated/figures/G02';
import { G03 } from '@/lib/generated/figures/G03';
import { G04 } from '@/lib/generated/figures/G04';
import { table } from '@/lib/generated/tables';

/**
 * Datasets (T114.2, T114.3; moved under About the Model by T127.3).
 *
 * Server component; the filterable record table is the one client island.
 *
 * ## The scope trap this section is built around
 *
 * T01 counts every file on disk. T02 counts labelled records per class. The
 * duration summary in T03 covers the supervised subset only. Those are three
 * different populations, and note.md records a table that once put 21.98 corpus
 * hours beside 3,240 supervised records on the same row — both numbers correct,
 * the row arithmetically impossible.
 *
 * So every block here states its own population in its heading or its caption,
 * and the tiles show the corpus count and the modelled count side by side.
 */
export function DatasetSection() {
  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Dataset Overview"
        lede="Four public phonocardiogram corpora, audited file by file rather than taken from their documentation. Where the published counts and the files on disk disagreed, the files won and the discrepancy is recorded."
        note={datasetSummary.scope_note}
      />

      <section>
        <SectionHeader
          eyebrow="Per corpus"
          title="Files on disk, and the subset actually modelled"
          level={2}
        />
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {datasetSummary.summary.map((row) => (
            <StatTile
              key={row.dataset_source}
              label={row.dataset_name}
              display={row.n_modelled_display}
              value={row.n_modelled}
              unit="modelled"
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

      <section>
        <SectionHeader eyebrow="T01" title="Dataset inventory" level={2} />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable
            table={table('T01')}
            hideColumns={['folder']}
            caption="Every file found on disk, with the subject-identifier origin per corpus. Counts here are the whole folder, not the modelled subset."
          />
        </GlassCard>
      </section>

      <section>
        <SectionHeader
          eyebrow="T02"
          title="Class distribution and imbalance"
          description="Labelled records only, per task. The five label spaces are separate and are never merged: a row belongs to exactly one of them."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T02')} />
        </GlassCard>

        <div className="mt-3 grid gap-3 lg:grid-cols-2">
          <GlassCard eyebrow="G02" title="Records per class">
            <GroupedBars
              source={G02}
              categoryColumn="class"
              valueColumns={['n_records']}
              label="Records per class, per dataset"
              caption="G02 — labelled records per class. Counts, not shares."
              height={340}
            />
            <FigureDownload figureId="G02" className="mt-2" />
          </GlassCard>
          <GlassCard eyebrow="G03" title="Class share within each dataset">
            <GroupedBars
              source={G03}
              categoryColumn="class"
              valueColumns={['share']}
              label="Class share within each dataset"
              caption="G03 — the same records as a share of their own dataset, which is what makes the imbalance comparable across corpora of very different sizes."
              height={340}
            />
            <FigureDownload figureId="G03" className="mt-2" />
          </GlassCard>
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="T03"
          title="Recording duration and sampling"
          description="Duration statistics over the supervised subset. Corpus-wide hours are in T01 and are a different population; the two are deliberately not shown on the same row."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T03')} />
        </GlassCard>
        <GlassCard className="mt-3" eyebrow="G04" title="Recording duration histogram">
          <GroupedBars
            source={G04}
            categoryColumn="bin_low_sec"
            valueColumns={['n_records']}
            label="Recording duration histogram"
            caption="G04 — duration histogram over supervised records. The bin totals here and the bars in G01 count different populations and do not sum to the same number."
            height={320}
          />
          <FigureDownload figureId="G04" className="mt-2" />
        </GlassCard>
      </section>

      <section>
        <SectionHeader
          eyebrow="Records"
          title="Every audited recording"
          description="Filter by corpus, subset, label or flag, and search by record or subject identifier. The whole audited corpus is loaded — nothing is sampled or truncated."
          level={2}
        />
        <div className="mt-4">
          <DatasetExplorer />
        </div>
      </section>
    </div>
  );
}
