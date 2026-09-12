import type { Metadata } from 'next';

import { EvidenceFilter } from '@/app/reports/EvidenceFilter';
import { ReportDownloads } from '@/app/reports/ReportDownloads';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { evidence } from '@/lib/generated/evidence';
import type { GeneratedEvidenceEntry } from '@/lib/generated/types';
import { prediction } from '@/lib/generated/prediction';
import { reports } from '@/lib/generated/reports';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Reports',
  description: 'Generated reports, and the evidence browser linking every displayed value to its source file.',
};

/**
 * Reports and the evidence browser (T117.3, T117.4).
 *
 * The evidence browser is rendered HERE, on the server, from
 * `generated/evidence`: one group per artifact, one row per exported column or
 * source file, each linking the served copy of the CSV it came from. The only
 * client code is a search box that hides non-matching groups, so the index is
 * in the HTML once and not a second time in a JavaScript bundle.
 */

function groupByArtifact(entries: GeneratedEvidenceEntry[]): [string, GeneratedEvidenceEntry[]][] {
  const groups = new Map<string, GeneratedEvidenceEntry[]>();
  for (const entry of entries) {
    const list = groups.get(entry.artifact) ?? [];
    list.push(entry);
    groups.set(entry.artifact, list);
  }
  return Array.from(groups.entries());
}

export default function Page() {
  const groups = groupByArtifact(evidence);

  return (
    <div className="space-y-6">
      <PageHeader title="Reports" lede={reports.note} />

      <section>
        <SectionHeader
          eyebrow="T117.3"
          title="Generate a report"
          description="Each document carries the screening-only disclaimer and the provenance of every number in it. Generating one needs the inference API running; the rest of this page does not."
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
          description="The six locked objectives, what answers each one, and what is still outstanding."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T29')} />
        </GlassCard>
      </section>

      <section id="evidence">
        <SectionHeader
          eyebrow="T117.4"
          title="Evidence browser"
          description="Every exported table column, figure and page payload, mapped to the committed file it was read from and that file's sha256 at export time. Each link opens the served copy of that file."
          level={2}
        />
        <div className="mt-4">
          <EvidenceFilter total={evidence.length} />
        </div>
        <div className="mt-3 space-y-1.5">
          {groups.map(([artifact, entries]) => {
            const files = Array.from(new Set(entries.map((entry) => entry.generated_from)));
            const text = (artifact + ' ' + files.join(' ') + ' ' + entries.map((e) => e.key).join(' ')).toLowerCase();
            return (
              <details
                key={artifact}
                id={'evidence-' + artifact}
                data-evidence-group=""
                data-evidence-text={text}
                data-evidence-count={String(entries.length)}
                className="scroll-mt-24 rounded-lg border border-line bg-panel px-3 py-2 open:shadow-panel"
              >
                <summary className="cursor-pointer text-body-md marker:text-accent">
                  <span className="font-mono text-label-md uppercase text-accent-strong">
                    {artifact}
                  </span>{' '}
                  <span className="text-body-sm text-ink-3">
                    {entries.length} {entries.length === 1 ? 'entry' : 'entries'} ·{' '}
                    {files.length === 1 ? files[0] : files.length + ' files'}
                  </span>
                </summary>
                <ul className="mt-2 space-y-1.5 border-t border-line pt-2 text-body-sm">
                  {entries.map((entry) => (
                    <li key={entry.key} className="grid gap-x-3 sm:grid-cols-[16rem_1fr]">
                      <span className="font-mono text-ink-2">{entry.key}</span>
                      <span className="break-all">
                        {entry.url ? (
                          <a href={entry.url} className="font-mono underline decoration-dotted underline-offset-2">
                            {entry.generated_from}
                          </a>
                        ) : (
                          <span className="font-mono">{entry.generated_from}</span>
                        )}{' '}
                        <span className="font-mono text-ink-3" title={entry.generated_from_sha256}>
                          sha256 {entry.generated_from_sha256.slice(0, 12)}
                        </span>
                        {entry.upstream_sources.length > 0 ? (
                          <span className="block text-ink-3">
                            built from {entry.upstream_sources.join(', ')}
                          </span>
                        ) : null}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            );
          })}
        </div>
      </section>
    </div>
  );
}
