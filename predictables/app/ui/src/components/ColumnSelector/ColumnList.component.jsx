import React from 'react';
import ColumnItem from './ColumnItem.component';
import './ColumnList.styles.css';

const ColumnList = ({ columns, selectedColumns, onColumnSelect }) => {
  return (
    <div className="column-list">
      {columns.map((column) => (
        <ColumnItem
          key={column}
          column={column}
          isSelected={selectedColumns.includes(column)}
          onColumnSelect={() => onColumnSelect(column)}
        />
      ))}
    </div>
  );
};

export default ColumnList;