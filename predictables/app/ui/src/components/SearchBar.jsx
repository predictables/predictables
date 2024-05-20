import React from 'react';
import './styles/SearchBar.css';

const SearchBar = ({ searchTerm, onSearchChange }) => {
  return (
    <input
      type="text"
      value={searchTerm}
      onChange={onSearchChange}
      placeholder="Search columns..."
      className="search-bar"
    />
  );
};

export default SearchBar;