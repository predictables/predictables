// DataSeries is similar to a series in pandas or polars. Single-column, single-typed data
// with `data`, `name`, `index`, `dtype` attributes

interface DataSeriesTypes {
  data: any[];
  name?: string;
  index?: any[];
  dtype?: 'string' | 'float' | 'integer' | 'category' | 'date' | 'any'
}

class DataSeries implements DataSeriesTypes {
  constructor(
    data: any[],
    name: string = '',
    index: any[] = [],
    dtype:  = 'any'): BaseDataSeries
}
