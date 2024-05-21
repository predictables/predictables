import React, { useState } from 'react';
import SearchBar from './SearchBar.component';
import ColumnList from './ColumnList.component';
import './SearchSelect.styles.css';

const SearchSelect = ({ columns }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedColumns, setSelectedColumns] = useState({ feature: [], target: [] });

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleColumnSelect = (column) => {
    // Logic to add/remove column from selectedColumns
    setSelectedColumns((prevSelected) => {
      const newSelected = { ...prevSelected };
      if (newSelected.feature.includes(column)) {
        newSelected.feature = newSelected.feature.filter((col) => col !== column);
      } else {
        newSelected.feature.push(column);
      }
      return newSelected;
    });
  };

  const filteredColumns = columns.filter((column) =>
    column.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="search-select">
      <SearchBar searchTerm={searchTerm} onSearchChange={handleSearchChange} />
      <ColumnList
        columns={filteredColumns}
        selectedColumns={selectedColumns.feature}
        onColumnSelect={handleColumnSelect}
      />
    </div>
  );
};

export default SearchSelect;