import React from 'react';
import { render, screen } from '@testing-library/react';
import ColumnList from './ColumnList.component';

test('renders ColumnList with correct number of ColumnItems', () => {
  const columns = ['Column1', 'Column2', 'Column3'];
  render(<ColumnList columns={columns} selectedColumns={[]} onColumnSelect={() => {}} />);
  const items = screen.getAllByText(/Column/i);
  expect(items.length).toBe(columns.length);
});