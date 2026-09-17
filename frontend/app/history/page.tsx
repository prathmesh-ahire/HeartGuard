import type { Metadata } from 'next';

import { HistoryBrowser } from '@/components/history/HistoryBrowser';
import { Reveal } from '@/components/motion/Reveal';
import { ButtonLink } from '@/components/ui/Button';
import { PageHeader } from '@/components/ui/PageHeader';
import { routeFor } from '@/lib/routes';

const route = routeFor('/history/');

export const metadata: Metadata = {
  title: 'History',
  description: route?.summary,
};

/**
 * History (T132). Every analysis saved by Analyse, read from the local History
 * store at runtime: it is the operator's own record, not a precomputed result,
 * so it cannot come from `generated/`. The page declares no number.
 */
export default function Page() {
  return (
    <div className="space-y-8">
      <Reveal>
        <PageHeader
          title="History"
          lede={route?.summary ?? ''}
          primaryAction={
            <ButtonLink href="/" tone="primary" icon="pulse">
              Analyse a recording
            </ButtonLink>
          }
        />
      </Reveal>
      <Reveal delay={0.05}>
        <HistoryBrowser />
      </Reveal>
    </div>
  );
}
