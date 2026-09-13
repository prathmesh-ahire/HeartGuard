import { AboutTabs } from '@/app/about/AboutTabs';

/**
 * About the Model (T127.3): the one place project and pipeline information
 * lives. Its tabs are sub-routes; see `AboutTabs`.
 */
export default function AboutLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="space-y-6">
      <header className="space-y-4">
        <h1 className="text-headline-lg text-ink">About the Model</h1>
        <AboutTabs />
      </header>
      {children}
    </div>
  );
}
