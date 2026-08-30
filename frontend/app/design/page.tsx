import type { Metadata } from 'next';

import { DesignClient } from '@/app/design/DesignClient';
import { EquationList } from '@/components/equations/Equations';
import { SectionHeader } from '@/components/ui/SectionHeader';

/**
 * The design reference (T111.6, extended by Phases 112 and 113).
 *
 * **This file is a server component and must stay one.** The equations render
 * through `katex.renderToString` at build time, which keeps 74 kB of KaTeX out
 * of the browser, renders the formulas with JavaScript disabled, and turns a
 * malformed formula into a failed build rather than red error text on a page
 * nobody re-checked. Marking this file `'use client'` -- or moving `EquationList`
 * into `DesignClient` -- silently undoes all three: the library follows the
 * component across the boundary and `scripts/20_check_bundle_budget.py` is what
 * caught it the first time.
 *
 * Everything with state lives in `DesignClient`.
 */
export const metadata: Metadata = {
  title: 'Design reference',
  description: 'Every design-system component in every state, for visual QA.',
};

export default function DesignPage() {
  return (
    <>
      <DesignClient />
      <section className="mt-10 space-y-4">
        <SectionHeader
          level={2}
          title="Equations"
          description={
            'The fifteen formulas of blueprint section 11, rendered by KaTeX at build ' +
            'time. Each names the module that implements it, verified at export: a ' +
            'formula here is a claim checked against the code, not a caption.'
          }
        />
        <EquationList />
      </section>
    </>
  );
}
