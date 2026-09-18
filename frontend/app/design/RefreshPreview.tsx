'use client';

import { useState, type ReactNode } from 'react';

import { PageTransition, Stagger, StaggerItem } from '@/components/motion/Reveal';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { GlassCard } from '@/components/ui/GlassCard';
import { Drawer, Modal, Sheet } from '@/components/ui/Overlay';
import { PageHeader } from '@/components/ui/PageHeader';
import {
  HistoryEmpty,
  HistoryLoading,
  InsightsEmpty,
  InsightsLoading,
  NoMatches,
  ServiceUnavailable,
} from '@/components/ui/PageStates';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { Skeleton, SkeletonChart, SkeletonList, SkeletonText, SkeletonTiles } from '@/components/ui/Skeleton';
import { cn } from '@/lib/cn';

/**
 * The Phase 128 primitives on the design reference (T128.6): every one, in
 * every state, in BOTH themes side by side.
 *
 * Each theme panel carries the `light` or `dark` class, which restates that
 * theme's tokens on the panel whatever the page theme is -- so one screen shows
 * both, and the overlays opened from a panel inherit that panel's theme.
 *
 * Nothing here is a result. The one number, the animated count, is the
 * recording count read from the generated payloads by `DesignClient`.
 */

const SCHEME = [
  { token: 'surface', role: 'Page background', swatch: 'bg-surface' },
  { token: 'accent', role: 'Accent', swatch: 'bg-accent' },
  { token: 'line', role: 'Rules and borders', swatch: 'bg-line' },
  { token: 'accent-soft', role: 'Soft fills', swatch: 'bg-accent-soft' },
] as const;

type Mode = 'light' | 'dark';
type OverlayKind = 'modal' | 'drawer' | 'sheet' | null;

function ThemePanel({ mode, children }: { mode: Mode; children: ReactNode }) {
  return (
    <div
      data-theme-preview={mode}
      className={cn(mode, 'min-w-0 space-y-8 rounded-2xl border border-line bg-surface p-5 text-ink')}
    >
      <p className="text-label-md uppercase text-accent">
        {mode === 'light' ? 'Light theme' : 'Dark theme'}
      </p>
      {children}
    </div>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-3">
      <p className="label-micro">{title}</p>
      {children}
    </div>
  );
}

function Primitives({ mode }: { mode: Mode }) {
  const [overlay, setOverlay] = useState<OverlayKind>(null);
  const close = () => setOverlay(null);

  return (
    <ThemePanel mode={mode}>
      <Block title="Scheme">
        <div className="grid grid-cols-2 gap-2">
          {SCHEME.map((entry) => (
            <div key={entry.token} className="overflow-hidden rounded-xl border border-line bg-panel">
              <div className={cn('h-10 w-full', entry.swatch)} />
              <p className="px-2 py-1.5 text-body-sm text-ink-2">
                {entry.role} <span className="font-mono text-ink-3">--{entry.token}</span>
              </p>
            </div>
          ))}
        </div>
      </Block>

      <Block title="Page title area">
        <div className="rounded-xl border border-line bg-panel p-5">
          <PageHeader
            level={2}
            eyebrow="Analyse"
            title="Analyse a recording"
            lede="One title, one lede, one primary action."
            actions={<Button tone="ghost">Samples</Button>}
            primaryAction={
              <Button tone="primary" icon="upload">
                Upload
              </Button>
            }
          />
        </div>
      </Block>

      <Block title="Controls and badges">
        <div className="flex flex-wrap gap-2">
          <Button tone="primary">Primary</Button>
          <Button tone="secondary">Secondary</Button>
          <Button tone="ghost">Ghost</Button>
          <Button tone="primary" disabled className="opacity-60">
            Disabled
          </Button>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone="neutral">neutral</Badge>
          <Badge tone="info">info</Badge>
          <Badge tone="good" dot>
            good
          </Badge>
          <Badge tone="warn" dot>
            warn
          </Badge>
          <Badge tone="danger" pulse>
            danger
          </Badge>
        </div>
      </Block>

      <Block title="Drawer, sheet and modal">
        <div className="flex flex-wrap gap-2">
          <Button data-testid={'open-modal-' + mode} onClick={() => setOverlay('modal')}>
            Open modal
          </Button>
          <Button data-testid={'open-drawer-' + mode} onClick={() => setOverlay('drawer')}>
            Open drawer
          </Button>
          <Button data-testid={'open-sheet-' + mode} onClick={() => setOverlay('sheet')}>
            Open sheet
          </Button>
        </div>
        <Modal
          open={overlay === 'modal'}
          onClose={close}
          title="Delete every record?"
          description="A decision gets a modal: short, centred, and it blocks the page until answered."
          footer={
            <>
              <Button tone="ghost" onClick={close}>
                Cancel
              </Button>
              <Button tone="primary" onClick={close}>
                Confirm
              </Button>
            </>
          }
        >
          <p className="text-body-lg text-ink-2">Escape, the close button and the scrim all close it.</p>
        </Modal>
        <Drawer
          open={overlay === 'drawer'}
          onClose={close}
          title="Record detail"
          description="The detail of one row, without leaving the list."
        >
          <SkeletonText lines={5} />
        </Drawer>
        <Sheet
          open={overlay === 'sheet'}
          onClose={close}
          title="Supporting detail"
          description="On a narrow screen, supporting content rises from the bottom."
        >
          <SkeletonText lines={3} />
        </Sheet>
      </Block>

      <Block title="Skeletons">
        <GlassCard>
          <Skeleton className="h-3 w-1/3" />
          <SkeletonText className="mt-4" />
        </GlassCard>
        <SkeletonTiles count={2} className="lg:grid-cols-2" />
        <SkeletonList rows={2} />
        <SkeletonChart />
      </Block>

      <Block title="Empty">
        <HistoryEmpty />
        <InsightsEmpty />
        <NoMatches onClear={() => undefined} />
      </Block>

      <Block title="Loading">
        <HistoryLoading />
        <InsightsLoading />
      </Block>

      <Block title="Error">
        <ServiceUnavailable onRetry={() => undefined} />
      </Block>
    </ThemePanel>
  );
}

export function RefreshPreview({
  count,
}: {
  count: { value: number | null; display: string } | null;
}) {
  const [replay, setReplay] = useState(0);

  return (
    <section className="space-y-6" data-refresh-preview="">
      <SectionHeader
        level={2}
        eyebrow="Design system refresh"
        title="Every new primitive, in both themes"
        description="The four-colour scheme, the page title area, the overlays, the skeletons and the empty, loading and error states of the new pages. Open an overlay from either panel; it takes that panel's theme."
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Primitives mode="light" />
        <Primitives mode="dark" />
      </div>

      <GlassCard
        eyebrow="Motion"
        title="Page transition, staggered reveal, animated number"
        meta={
          <Button size="sm" onClick={() => setReplay((value) => value + 1)}>
            Replay
          </Button>
        }
      >
        <p className="max-w-reading text-body-lg text-ink-2">
          With reduced motion turned on, each of these appears in its final state without
          moving.
        </p>
        <PageTransition key={'page-' + replay}>
          <Stagger key={'stagger-' + replay} className="mt-5 grid gap-3 sm:grid-cols-3">
            {['First', 'Second', 'Third'].map((label) => (
              <StaggerItem key={label}>
                <div className="rounded-xl border border-line bg-sunken p-4 text-body-lg text-ink">
                  {label}
                </div>
              </StaggerItem>
            ))}
          </Stagger>
        </PageTransition>
        {count ? (
          <p className="mt-5 text-body-lg text-ink-2">
            Recordings on disk:{' '}
            <AnimatedCounter key={'count-' + replay} value={count.value} display={count.display} />
          </p>
        ) : null}
      </GlassCard>
    </section>
  );
}
