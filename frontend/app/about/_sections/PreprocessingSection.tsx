import { SignalExplorer } from '@/app/about/_sections/SignalExplorer';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { ResultsTable } from '@/components/table/ResultsTable';
import { Disclosure } from '@/components/ui/Disclosure';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { table } from '@/lib/generated/tables';

/**
 * Signal Preprocessing (T114.4, T114.5; moved under About the Model by T127.3).
 *
 * Server component. The record picker and the stage toggles are the client
 * island; every waveform they switch between was computed in Python.
 *
 * The spectrogram and the wavelet decomposition are served as their canonical
 * 300 dpi figures rather than redrawn in the browser. Both exceed the exporter's
 * inline budget — G06 is a 514 x 47 grid and G09 is 12,032 rows — and shipping
 * them as JSON would put roughly a megabyte into this route to reproduce a
 * picture that already exists.
 */
export function PreprocessingSection() {
  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
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

      <section>
        <SectionHeader
          eyebrow="Interactive"
          title="What each stage does, on a real recording"
          description="Pick a recording, then switch the filter and normalization stages on and off. All four combinations were computed in advance by the pipeline itself; nothing is filtered in your browser."
          level={2}
        />
        <div className="mt-4">
          <SignalExplorer />
        </div>
      </section>

      <section>
        <SectionHeader eyebrow="T04" title="Preprocessing configuration" level={2} />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable
            table={table('T04')}
            caption="The settings every record was processed with, read from the pipeline configuration rather than restated. Two runs of the same command produce identical numbers only because these are fixed."
          />
        </GlassCard>
      </section>

      <section>
        <SectionHeader eyebrow="Detail" title="Full-resolution figures" level={2} />
        <div className="mt-4 space-y-3">
          <Disclosure
            summary={
              <>
                <span className="mr-2 text-label-sm uppercase text-ink-3">G05</span>
                Before and after filtering, at full resolution
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              The interactive view above is strided to a display budget. This figure is the same
              comparison at full sample resolution, rendered at 300 dpi.
            </p>
            <GlassCard eyebrow="G05" title="Filtered against raw, full resolution">
              <FigureDownload figureId="G05" />
            </GlassCard>
          </Disclosure>

          <Disclosure
            summary={
              <>
                <span className="mr-2 text-label-sm uppercase text-ink-3">G06 / G09</span>
                Spectrogram and wavelet decomposition, normal versus abnormal
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              Two spectrogram panels on one shared colour scale, so the two are directly
              comparable (the underlying grid is 514 frequency bins across 47 time frames). G09 is
              the wavelet decomposition of the same signal; its sub-bands span three orders of
              magnitude, so each is drawn on its own y-axis — a shared axis would render the
              detail bands as flat lines and imply they carry nothing.
            </p>
            <div className="grid gap-3 lg:grid-cols-2">
              <GlassCard eyebrow="G06" title="Spectrogram, shared colour scale">
                <FigureDownload figureId="G06" />
              </GlassCard>
              <GlassCard eyebrow="G09" title="Wavelet decomposition">
                <FigureDownload figureId="G09" />
              </GlassCard>
            </div>
          </Disclosure>
        </div>
      </section>
    </div>
  );
}
