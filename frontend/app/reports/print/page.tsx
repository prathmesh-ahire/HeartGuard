import type { Metadata } from 'next';

import { PrintView } from '@/app/reports/print/PrintView';

export const metadata: Metadata = {
  title: 'Print report',
  description: 'A print-friendly view of one analysis, opened from the Reports page.',
};

/** Print-friendly single-result view (T134.3). */
export default function Page() {
  return <PrintView />;
}
