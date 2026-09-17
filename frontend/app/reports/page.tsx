import type { Metadata } from 'next';

import { Reveal } from '@/components/motion/Reveal';
import { ReportsBoard } from '@/components/reports/ReportsBoard';
import { PageHeader } from '@/components/ui/PageHeader';
import { routeFor } from '@/lib/routes';

const route = routeFor('/reports/');

export const metadata: Metadata = {
  title: 'Reports',
  description: route?.summary,
};

/**
 * Reports (T117.3; trimmed by T127.1; rebuilt in Phase 134).
 *
 * The evidence browser and the thesis-era experiment/objective DOCX pickers
 * that used to sit here are gone: they named internal experiment ids and
 * objective codes, which is exactly the audience Part XII moved away from.
 * This page is now the operator's own report generator -- built entirely
 * from what the operator has actually analysed (History) plus the one
 * always-available model summary, never from thesis provenance.
 */
export default function Page() {
  return (
    <div className="space-y-8">
      <Reveal>
        <PageHeader title="Reports" lede={route?.summary ?? ''} />
      </Reveal>
      <Reveal delay={0.05}>
        <ReportsBoard />
      </Reveal>
    </div>
  );
}
