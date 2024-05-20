import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SearchBar from '../src/components/SearchBar';

test('renders SearchBar with correct initial value', () => {
  render(<SearchBar searchTerm="test" onSearchChange={() => {}} />);
  const input = screen.getByPlaceholderText('Search columns...');
  expect(input.value).toBe('test');
});

test('calls onSearchChange when input value changes', () => {
  const onSearchChange = jest.fn();
  render(<SearchBar searchTerm="" onSearchChange={onSearchChange} />);
  const input = screen.getByPlaceholderText('Search columns...');
  fireEvent.change(input, { target: { value: 'new value' } });
  expect(onSearchChange).toHaveBeenCalledTimes(1);
});