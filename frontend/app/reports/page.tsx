import type { Metadata } from 'next';

import { ReportDownloads } from '@/app/reports/ReportDownloads';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { prediction } from '@/lib/generated/prediction';
import { reports } from '@/lib/generated/reports';
import { table } from '@/lib/generated/tables';
import { routeFor } from '@/lib/routes';

export const metadata: Metadata = {
  title: 'Reports',
  description: routeFor('/reports/')?.summary,
};

/**
 * Reports (T117.3; trimmed by T127.1, rebuilt in Phase 134).
 *
 * The evidence browser that used to sit here listed repository file paths and
 * digests. That traceability is still enforced -- the evidence index is checked on
 * every build by the displayed-value audit -- but it is not something a person
 * using the app reads, so it is no longer rendered.
 */
export default function Page() {
  return (
    <div className="space-y-8">
      <PageHeader title="Reports" lede={routeFor('/reports/')?.summary ?? ''} />

      <section>
        <SectionHeader
          eyebrow="Download"
          title="Generate a report"
          description="Generating a report needs the analysis service running; the rest of this page does not."
          level={2}
        />
        <div className="mt-4">
          <ReportDownloads reports={reports} samples={prediction.samples} />
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="T29"
          title="Objective coverage"
          description="The six research objectives, what answers each one, and what is still outstanding."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T29')} hideColumns={['modules', 'evidence_files']} />
        </GlassCard>
      </section>
    </div>
  );
}
