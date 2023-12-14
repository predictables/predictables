'use client';

import React, { useContext } from 'react';
import DataTableContext from '@app/_context/DataTableContext';
import Table from '@components/Table';

const ViewDataPage = () => {
  const context = useContext(DataTableContext);
  if (!context) {
    return <div>Context is null</div>;
  } else {
    const { dt } = context;
    return (
      <div>
        <Table dt={dt} />
      </div>
    );
  }
};

export default ViewDataPage;
