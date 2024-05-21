import React from 'react';
import './ColumnItem.styles.css';w

const ColumnItem = ({ column, isSelected, onColumnSelect }) => {
  return (
    <div
      className={`column-item ${isSelected ? 'selected' : ''}`}
      onClick={onColumnSelect}
    >
      {column}
    </div>
  );
};

export default ColumnItem;