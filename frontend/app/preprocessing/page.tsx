import type { Metadata } from 'next';

import { SignalExplorer } from '@/app/preprocessing/SignalExplorer';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { figure } from '@/lib/generated/figures';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Signal Preprocessing',
  description:
    'What the pipeline does to a recording before a single feature is computed, shown on real recordings.',
};

/**
 * Signal Preprocessing (T114.4, T114.5).
 *
 * Server component. The record picker and the stage toggles are the client
 * island; every waveform they switch between was computed in Python.
 *
 * The spectrogram and the wavelet decomposition are served as their canonical
 * 300 dpi figures rather than redrawn in the browser. Both exceed the exporter's
 * inline budget — G06 is a 514 x 47 grid and G09 is 12,032 rows — and shipping
 * them as JSON would put roughly a megabyte into this route to reproduce a
 * picture that already exists. The figure and its source CSV are both linked, so
 * the underlying numbers remain reachable.
 */
export default function Page() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Signal Preprocessing"
        lede={
          <>
            Every recording passes through the same six steps before a feature is
            computed: load, collapse to mono, resample to the working rate, measure
            quality, band-pass filter, normalize.
          </>
        }
        note="The corpora arrive at 2 kHz, 4 kHz and 44.1 kHz, so resampling is not a formality — it is what makes a feature computed on a PASCAL A recording comparable to the same feature on a PhysioNet one."
      />

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T114.4 / T114.5"
          title="What each stage does, on a real recording"
          description="Pick a recording, then switch the filter and normalization stages on and off. The four combinations were all computed by the pipeline's own functions and exported; nothing is filtered in your browser."
          level={2}
        />
        <div className="mt-4">
          <SignalExplorer />
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="T04" title="Preprocessing configuration" level={2} />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable
            table={table('T04')}
            caption="The settings every record was processed with, read from the pipeline configuration rather than restated. Two runs of the same command produce identical numbers only because these are fixed."
          />
        </GlassCard>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="G05"
          title="Before and after filtering, at full resolution"
          description="The interactive view above is strided to a display budget. This figure is the same comparison at full sample resolution, rendered at 300 dpi."
          level={2}
        />
        <GlassCard className="mt-4" eyebrow="G05" title="Filtered against raw, full resolution">
          <FigureDownload figureId="G05" />
        </GlassCard>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="G06"
          title="Spectrogram, normal versus abnormal"
          description="Two panels on one shared colour scale, so the two are directly comparable. Served as the canonical figure: the underlying grid is 514 frequency bins across 47 time frames and does not belong in a page bundle."
          level={2}
        />
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          <GlassCard eyebrow="G06" title="Spectrogram, shared colour scale">
            <FigureDownload figureId="G06" />
          </GlassCard>
          <GlassCard eyebrow="G09" title="Wavelet decomposition">
            <FigureDownload figureId="G09" />
          </GlassCard>
        </div>
        <p className="mt-3 text-sm text-ink-2">
          G09 is the wavelet decomposition of the same signal. Its sub-bands span three
          orders of magnitude, so each is drawn on its own y-axis — a shared axis would
          render the detail bands as flat lines and imply they carry nothing.
        </p>
      </section>

      {/* ----------------------------------------------------------------- */}
      <GlassCard as="section" eyebrow="Provenance" bodyClassName="text-body-sm text-ink-2">
        <p>
          Figures on this page were generated by{' '}
          <span className="font-mono">scripts/19_data_graphs.py</span> and are registered
          in the figure registry with the sha256 of the CSV each was plotted from. The
          interactive waveforms come from{' '}
          <span className="font-mono">outputs/02_preprocessing/preprocessing_examples.csv</span>,
          which is regenerated from the corpus whenever the corpus is present.
        </p>
        <p className="mt-2">
          {figure('G05')?.title ?? 'G05'} and{' '}
          {figure('G06')?.title ?? 'G06'} are not inlined as data by design; their full
          numeric content is in{' '}
          <span className="font-mono">outputs/13_figures_diagrams/</span>.
        </p>
      </GlassCard>
    </div>
  );
}
