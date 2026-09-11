import { cn } from '@/lib/cn';

/**
 * A link from a displayed value to the file it was read from (T117.4).
 *
 * The exporter copies every table and figure CSV it read into
 * `public/evidence/<same path>`, so a repository-relative `outputs/...` path
 * maps to a URL the static site actually serves. A path outside `outputs/` has
 * no served copy and renders as plain text rather than as a link to a 404.
 */
export function evidenceUrl(path: string): string | null {
  return path.startsWith('outputs/') ? '/evidence/' + path : null;
}

/** The anchor of an artifact's group in the evidence browser on /reports/. */
export function evidenceAnchor(artifact: string): string {
  return '/reports/#evidence-' + artifact;
}

export function EvidenceLink({
  path,
  artifact,
  className,
}: {
  path: string;
  /** When given, also links the artifact's entry in the evidence browser. */
  artifact?: string;
  className?: string;
}) {
  const url = evidenceUrl(path);
  return (
    <span className={cn('font-mono break-all', className)}>
      {url === null ? (
        path
      ) : (
        <a href={url} className="underline decoration-dotted underline-offset-2 hover:decoration-solid">
          {path}
        </a>
      )}
      {artifact ? (
        <>
          {' · '}
          <a
            href={evidenceAnchor(artifact)}
            className="underline decoration-dotted underline-offset-2 hover:decoration-solid"
          >
            evidence
          </a>
        </>
      ) : null}
    </span>
  );
}
