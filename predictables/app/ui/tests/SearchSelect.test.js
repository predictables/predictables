import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SearchSelect from '../src/components/SearchSelect';

const columns = ['Column1', 'Column2', 'Column3'];

test('updates search term when SearchBar changes', () => {
  render(<SearchSelect columns={columns} />);
  const input = screen.getByPlaceholderText('Search columns...');
  fireEvent.change(input, { target: { value: 'Col' } });
  expect(input.value).toBe('Col');
});

test('filters columns correctly based on search term', () => {
  render(<SearchSelect columns={columns} />);
  const input = screen.getByPlaceholderText('Search columns...');
  fireEvent.change(input, { target: { value: 'Column1' } });
  const items = screen.getAllByText(/Column/i);
  expect(items.length).toBe(1);
  expect(items[0].textContent).toBe('Column1');
});

test('updates selected columns when ColumnItem is clicked', () => {
  render(<SearchSelect columns={columns} />);
  const item = screen.getByText('Column1');
  fireEvent.click(item);
  expect(item).toHaveClass('selected');
});