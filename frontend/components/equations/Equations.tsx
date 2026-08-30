import 'katex/dist/katex.min.css';

import katex from 'katex';

import { cn } from '@/lib/cn';
import { equations as generatedEquations, type GeneratedEquation } from '@/lib/generated';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The fifteen equations of blueprint section 11, rendered with KaTeX (T113.5).
 *
 * ## Rendered at build time, not in the browser
 *
 * This is a **server component**: `katex.renderToString` runs during the static
 * export and the page ships finished HTML. Three things follow. KaTeX's ~70 kB
 * of JavaScript never reaches the client. The equations render with JavaScript
 * disabled, which matters for a document somebody will print. And a malformed
 * formula fails the **build** rather than rendering as red error text on a page
 * nobody re-checked -- `throwOnError` is left on for exactly that reason.
 *
 * ## The LaTeX is generated, like everything else
 *
 * It comes from `generated/equations.json`, which `src/reporting/equations.py`
 * emits after verifying that each equation names a module that exists and a
 * symbol that appears in it. A formula on this page is a claim the build
 * checked against the implementation, which is what T100.5 will ask for in
 * Phase 100.
 */

export function Equation({ latex, className }: { latex: string; className?: string }) {
  const html = katex.renderToString(latex, {
    displayMode: true,
    throwOnError: true,
    strict: 'warn',
  });
  return (
    <div
      className={cn('overflow-x-auto py-1', className)}
      // The input is our own generated LaTeX and KaTeX's output is sanitised
      // markup; `trust` is left at its default false, so \htmlData and \url
      // are refused even if a formula ever tried them.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/** Inline maths, for a symbol named in running text. */
export function InlineMath({ latex }: { latex: string }) {
  const html = katex.renderToString(latex, {
    displayMode: false,
    throwOnError: true,
    strict: 'warn',
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function EquationCard({ equation }: { equation: GeneratedEquation }) {
  return (
    <article className={cn(SURFACE.card, 'p-5')} id={'equation-' + equation.key}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className={cn(TYPE_SCALE.h3)}>
          <span className={cn(SURFACE.subtle, 'mr-2 tabular-nums')}>{equation.number}.</span>
          {equation.name}
        </h3>
        <span className={cn(TYPE_SCALE.caption, SURFACE.muted)}>{equation.use}</span>
      </header>

      <Equation latex={equation.latex} className="mt-3" />

      <dl className={cn(TYPE_SCALE.caption, 'mt-3 space-y-1')}>
        {equation.symbols.map((symbol) => (
          <div key={symbol.symbol} className="flex flex-wrap gap-2">
            <dt className="min-w-16">
              <InlineMath latex={symbol.symbol} />
            </dt>
            <dd className={SURFACE.muted}>{symbol.meaning}</dd>
          </div>
        ))}
      </dl>

      {equation.transcription_note === null ? null : (
        <p
          className={cn(
            TYPE_SCALE.caption,
            'mt-3 border-l-2 border-amber-400 pl-3 dark:border-amber-600',
          )}
        >
          <span className="font-medium">Transcription note. </span>
          {equation.transcription_note}
        </p>
      )}

      <p className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'mt-3 font-mono')}>
        implemented in {equation.implemented_in} · {equation.implements}
      </p>
    </article>
  );
}

export function EquationList({ className }: { className?: string }) {
  return (
    <div className={cn('space-y-4', className)}>
      <p className={cn(TYPE_SCALE.caption, SURFACE.subtle)}>
        {generatedEquations.n_equations} equations from {generatedEquations.source}.{' '}
        {generatedEquations.note}
      </p>
      {generatedEquations.equations.map((equation) => (
        <EquationCard key={equation.key} equation={equation} />
      ))}
    </div>
  );
}
