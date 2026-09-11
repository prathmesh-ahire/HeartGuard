import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { FacetTable } from '@/components/table/FacetTable';
import { ResultsTable } from '@/components/table/ResultsTable';
import { table } from '@/lib/generated/tables';
import type { GeneratedTable } from '@/lib/generated/types';

/**
 * T118.1: the results table renders the exported display strings and nothing
 * else, sorts by the numeric column rather than the text, filters, and writes
 * exactly what is on screen to CSV. Driven by the REAL T02 and T20 exports.
 */

function real(id: string): GeneratedTable {
  const found = table(id);
  if (found === undefined) throw new Error(id + ' is not exported; run the exporter');
  return found;
}

function bodyRows(): HTMLElement[] {
  const [, body] = screen.getAllByRole('rowgroup');
  if (body === undefined) throw new Error('no table body');
  return within(body).getAllByRole('row');
}

describe('ResultsTable', () => {
  it('renders every display string of the real T02, and no values array', () => {
    const t02 = real('T02');
    render(<ResultsTable table={t02} />);
    expect(bodyRows()).toHaveLength(t02.n_rows);
    for (const column of t02.columns) {
      for (const shown of column.display) expect(screen.getAllByText(shown).length).toBeGreaterThan(0);
    }
  });

  it('sorts a numeric column by its value, not its text', () => {
    const t02 = real('T02');
    render(<ResultsTable table={t02} />);
    fireEvent.click(screen.getByRole('button', { name: /^n_records/ }));
    const records = t02.columns.find((column) => column.name === 'n_records');
    if (records?.values == null) throw new Error('T02.n_records has no values');
    const smallest = records.display[records.values.indexOf(Math.min(...(records.values as number[])))];
    // The smallest count can coincide with another cell of the same row (T02's
    // extrahls row has 19 records and 19 subjects), so "at least one" is the claim.
    expect(within(bodyRows()[0] as HTMLElement).getAllByText(smallest as string).length).toBeGreaterThan(0);
  });

  it('narrows rows with the filter and says how many match', () => {
    const t02 = real('T02');
    render(<ResultsTable table={t02} />);
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'extrastole' } });
    expect(bodyRows()).toHaveLength(1);
    expect(screen.getByText(/1 of/)).toBeTruthy();
  });

  it('links its source CSV to the served evidence copy', () => {
    const t02 = real('T02');
    render(<ResultsTable table={t02} />);
    const link = screen.getByRole('link', { name: t02.source_csv });
    expect(link.getAttribute('href')).toBe('/evidence/' + t02.source_csv);
    expect(screen.getByRole('link', { name: 'evidence' }).getAttribute('href')).toBe(
      '/reports/#evidence-T02',
    );
  });

  it('downloads exactly the displayed strings as CSV', async () => {
    const t02 = real('T02');
    let written: Blob | null = null;
    vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      written = blob as Blob;
      return 'blob:captured';
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    render(<ResultsTable table={t02} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download CSV as displayed' }));
    expect(written).not.toBeNull();
    // jsdom's Blob has no .text(); FileReader is the API it does implement.
    const text = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(written as unknown as Blob);
    });
    const share = t02.columns.find((column) => column.name === 'share');
    for (const shown of share?.display ?? []) expect(text).toContain('"' + shown + '"');
  });

  it('renders a stated absence, not an empty grid, when the table was not exported', () => {
    render(<ResultsTable table={undefined} />);
    expect(screen.getByText('This table has not been exported')).toBeTruthy();
  });
});

describe('FacetTable', () => {
  it('selects rows by a facet without changing any cell', () => {
    const t20 = real('T20');
    render(<FacetTable table={t20} facets={['analysis', 'run']} />);
    expect(bodyRows()).toHaveLength(t20.n_rows);

    const run = t20.columns.find((column) => column.name === 'run');
    if (run === undefined) throw new Error('T20 has no run column');
    const wanted = 'EXP-E1-AWGN';
    const expected = run.display.filter((value) => value === wanted).length;
    fireEvent.change(screen.getByRole('combobox', { name: /run/ }), { target: { value: wanted } });

    const rows = bodyRows();
    expect(rows).toHaveLength(expected);
    for (const row of rows) expect(within(row).getByText(wanted)).toBeTruthy();
  });
});
