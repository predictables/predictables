import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ColumnItem from './ColumnItem.component';

test('renders ColumnItem with correct text', () => {
  const column = 'Column1';
  render(<ColumnItem column={column} isSelected={false} onColumnSelect={() => {}} />);
  const item = screen.getByText(column);
  expect(item).toBeInTheDocument();
});

test('calls onColumnSelect when ColumnItem is clicked', () => {
  const column = 'Column1';
  const onColumnSelect = jest.fn();
  render(<ColumnItem column={column} isSelected={false} onColumnSelect={onColumnSelect} />);
  const item = screen.getByText(column);
  fireEvent.click(item);
  expect(onColumnSelect).toHaveBeenCalledTimes(1);
});