'use client';

import { useRef, useEffect, useState } from 'react';

import { ScatterProps } from './interfaces';
import Point from './_primatives/Point';
import DataTable from '@models/DataTable/DataTable';
import DataSeries from '@models/DataTable/DataSeries';

import * as d3 from 'd3';

const Scatter = ({
  height = 500,
  width = 750,
  data,
  xCol,
  yCol,
  xScaleType = 'linear',
  yScaleType = 'linear',
  colorCol = '',
  sizeCol = '',
  marginTop = 20,
  marginRight = 20,
  marginLeft = 40,
  marginBottom = 30,
}: ScatterProps) => {
  const [x, setX] = useState<DataSeries>(DataSeries.placeholderDataSeries());
  const [y, setY] = useState<DataSeries>(DataSeries.placeholderDataSeries());
  const [isDataLoaded, setIsDataLoaded] = useState<boolean>(false);

  useEffect(() => {
    if (isDataLoaded) {
      setX(data?.col(xCol));
      setY(data?.col(yCol));
      setIsDataLoaded(true);
    } else {
    }
  }, [data, xCol, yCol, isDataLoaded]);

  const xDomain: number[] = x.dataRange();
  const yDomain: number[] = y.dataRange();
  const xRange: number[] = [marginLeft, width - marginRight];
  const yRange: number[] = [height - marginBottom, marginTop];

  const xAxis: any = d3.scaleLinear(xDomain, xRange);
  const yAxis: any = d3.scaleLinear(yDomain, yRange);

  const gxAxis: any = useRef();
  const gyAxis: any = useRef();

  useEffect(
    () => void d3.select(gxAxis.current).call(d3.axisBottom(xAxis)),
    [gxAxis, xAxis],
  );
  useEffect(
    () => void d3.select(gyAxis.current).call(d3.axisLeft(yAxis)),
    [gyAxis, yAxis],
  );

  return (
    <g>
      {isDataLoaded ? (
        <>
          <g
            ref={gxAxis}
            transform={`translate(0, ${height - marginBottom})`}
          />
          <g ref={gyAxis} transform={`translate(${marginLeft}, 0)`} />
          <g>
            {x.data().map((_: any, i: number) => {
              let tempX = x.data(i);
              let tempY = y.data(i);
              console.log('tempX:', tempX);
              console.log('tempY:', tempY);
              <Point
                key={i}
                x={xAxis(tempX)}
                y={yAxis(tempY)}
                radius={5}
                edgeColor="black"
                fillColor="red"
              />;
            })}
          </g>
        </>
      ) : (
        <div>Hi</div>
      )}
    </g>
  );
};

export default Scatter;
