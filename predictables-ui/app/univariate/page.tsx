'use client';

import React, { useContext, useState, useEffect } from 'react';
import Heading from '@components/Heading';
import PlotContainer from '@components/plotting/PlotContainer';
import Point from '@components/plotting/_primatives/Point';
import Scatter from '@components/plotting/Scatter';
import DataTableContext from '@app/_context/DataTableContext';
import DataTable from '@models/DataTable/DataTable';

const UnivariatePage = () => {
  // const dt: DataTable | null = useContext(DataTableContext);
  const singlePlotHeight = 500;
  const singlePlotWidth = 750;
  const xCol: string = 'mean_area';
  const yCol: string = 'worst_concave_points';

  const [dt, setDT] = useState(useContext(DataTableContext));

  return (
    <section className="items-center h-[100vh] flex flex-col">
      <Heading text="Univariate" />
      <PlotContainer height={singlePlotHeight} width={singlePlotWidth}>
        <Scatter
          height={singlePlotHeight}
          width={singlePlotWidth}
          data={dt?.dt}
          xCol={xCol}
          yCol={yCol}
        />
      </PlotContainer>
    </section>
  );
};

export default UnivariatePage;
