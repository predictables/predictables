// DataTable class is currently an array of DataSeries objects, with some additional methods
// Particular interest in the .render() method, which returns a formatted HTML table
import DataSeries, { dtypeTypes } from './DataSeries';

interface DataTableProps {
  values: DataSeries[];
  columns?: string[];
  index?: string[];
}

/**
 * @class DataTable
 * @description A class representing a table of data
 * @param {DataSeries[]} values - An array of DataSeries objects
 * @param {string[]} columns - An array of column names
 * @param {string[]} index - An array of index values
 * @property {DataSeries[]} values - An array of DataSeries objects
 * @property {string[]} columns - An array of column names
 * @property {string[]} index - An array of index values
 *
 * @method {height} - Returns the height of the DataTable
 * @method {width} - Returns the width of the DataTable
 * @method {nRows} - Returns the number of rows in the DataTable
 * @method {nCols} - Returns the number of columns in the DataTable
 * @method {at} - Returns the value at the given row and column
 */
class DataTable {
  values: DataSeries[];
  columns: string[];
  index: string[];

  constructor(values: DataSeries[], columns: string[] = [], index?: string[]) {
    // ensure each element of the values array is a DataSeries
    if (values.length === 0) {
      throw new Error('Values array must not be empty');
    }

    // ensure all values are DataSeries
    if (values.some((v) => !(v instanceof DataSeries))) {
      throw new Error('Values array must contain only DataSeries objects');
    }

    // ensure all values are the same length
    const lengths = values.map((v) => v.length);
    if (lengths.some((l) => l !== lengths[0])) {
      throw new Error('All DataSeries must be the same length');
    }

    // if an index was not passed, ensure all values have the same index array, and if not, reset all of them to the default
    if (index === null) {
      const indices = values.map((v) => v.index);

      // test if all indices are elementwise equal
      const cond = (i, j) => i === j;
      const indicesEqual = indices.every((i) =>
        indices.every((j) => cond(i, j)),
      );

      // if they are not all equal, reset them all to the default, which is the index of the first value
      if (!indicesEqual) {
        const defaultIndex = values[0].index;
        values.forEach((v) => (v.index = defaultIndex));
      }

      // set the index to the index of the first value
      this.index = values[0].index;
    } else {
      // set every index to be the passed index, as well as the class index
      values.forEach((v: DataSeries) => (v.index = index));
      this.index = index;
    }

    // if a columns array was not passed, set it to the names of the values
    if (columns.length === 0) {
      values.forEach((v, i) => {
        // ensure all values have a name -- if not give them a default name (column0, column1, etc)
        if (v.name === null || v.name === '') {
          v.name = `column${i}`;
        }
      });

      // set the columns to the names of the values
      this.columns = values.map((v) => v.name);

      // set the values array
      this.values = values;
    } else {
      // ensure the columns array is the same length as the values array
      if (columns.length !== values.length) {
        throw new Error(
          'Columns array must be the same length as the values array',
        );
      }
    }

    // for each column, give the class a property with the column name that returns the DataSeries
    this.columns.forEach((c, i) => {
      this[c] = values[i];
    });

    // set the values array
    this.values = values;

    // set the shape
    this.shape = [this.width, this.height];
  }
  get height() {
    return this.values[0].length;
  }

  get width() {
    // Returns the width of the DataTable
    return this.values.length;
  }

  get nRows() {
    // Returns the number of rows in the DataTable
    return this.height;
  }

  get nCols() {
    // Returns the number of columns in the DataTable
    return this.width;
  }

  /**
   * @method {col}
   * @description Returns a single column by name as a DataSeries
   * @param {number | string} column - Either the column number or a string representing the name of the DataSeries
   * @returns {DataSeries} - A single column as a DataSeries
   */
  col(column: number | string): DataSeries {
    // Returns a single column by name as a DataSeries
    const columns: string[] = this.columns;
    const index = columns.indexOf(column.toString());
    if (index === -1) {
      throw new Error(`Column ${column} not found`);
    }
    return this.values[index];
  }

  /**
   * @method {df}
   * @description Returns a copy of the DataTable
   * @returns {DataTable} - A copy of the DataTable
   */
  df(): DataTable {
    const values: DataSeries[] = this.values.map((v) => v.copy());
    const columns: string[] = [...this.columns];
    const index: string[] = [...this.index];
    return new DataTable(values, columns, index);
  }

  /**
   * @method {at}
   * @description Returns the value at the given row and column
   * @param {number | string} row - Either the row number or a string representing the name of the index
   * @param {number | string} column - Either the column number or a string representing the name of the DataSeries
   * @returns {any} - The value at the given row and column. Can be any type that can be held inside a DataSeries.
   */
  at(row: number | string, column: number | string): any {
    // Returns the value at the given row and column
    const rowIndex = this.index.indexOf(row.toString());
    if (rowIndex === -1) {
      throw new Error(`Row ${row} not found`);
    } else {
      return this.values[this.columns.indexOf(column.toString())].values[
        rowIndex
      ];
    }
  }

  /**
   * @method {toJSON}
   * @description Returns a JSON representation of the DataTable
   * @param {boolean} byRow - Whether to return the JSON by row or by column. Default is false, which returns by column.
   * @returns {any} - A JSON representation of the DataTable
   * @example
   * const dt = new DataTable([
   *    DataSeries.fromArray([1, 2, 3], name="col1"),
   *    DataSeries.fromArray([4, 5, 6], name="col2"),
   *    DataSeries.fromArray([7, 8, 9], name="col3"),
   * ]);
   * dt.toJSON();
   * > {
   * >    "col1": [1, 2, 3],
   * >    "col2": [4, 5, 6],  
   * >    "col3": [7, 8, 9],
   * > }
   */
  toJSON(byRow: boolean = false): any {
    // Returns a JSON representation of the DataTable
    const json = {};
    if (byRow) {
      this.index.forEach((i) => {
        json[i] = {};
        this.columns.forEach((c) => {
          json[i][c]:any = this.at(i, c);
        });
      });
    } else {
      this.columns.forEach((c) => {
        json[c] = {};
        this.index.forEach((i) => {
          json[c][i]:any = this.at(i, c);
        });
      });
    }
  }

  /**
   * @method {head}
   * @description Returns the first n rows of the DataTable
   * @param {number} n - The number of rows to return. Default is 5.
   * @returns {DataTable} - The first n rows of the DataTable
   * @example
   * const dt = new DataTable([
   *   DataSeries.fromArray([1, 2, 3, 4, 5], name="col1"),
   *   DataSeries.fromArray([6, 7, 8, 9, 10], name="col2"),
   *   DataSeries.fromArray([11, 12, 13, 14, 15], name="col3"),
   * ]);
   * dt.head(3).toJSON();
   * > {
   * >  "col1": [1, 2, 3],
   * >  "col2": [6, 7, 8],  
   * >  "col3": [11, 12, 13],
   * > }
   */
  head(n:number = 5) {
    const values = this.values.map((v) => v.head(n));
    return new DataTable(values, this.columns, this.index);
  }

  /**
   * @method {tail}
   * @description Returns the last n rows of the DataTable
   * @param {number} n - The number of rows to return. Default is 5.
   * @returns {DataTable} - The last n rows of the DataTable
   * @example
   * const dt = new DataTable([
   *   DataSeries.fromArray([1, 2, 3, 4, 5], name="col1"),
   *   DataSeries.fromArray([6, 7, 8, 9, 10], name="col2"),
   *   DataSeries.fromArray([11, 12, 13, 14, 15], name="col3"),
   * ]);
   * dt.tail(2).toJSON();
   * > {
   * >  "col1": [4, 5],
   * >  "col2": [9, 10],  
   * >  "col3": [14, 15],
   * > }
   */
  tail(n:number = 5) {
    const values = this.values.map((v) => v.tail(n));
    return new DataTable(values, this.columns, this.index);
  }

  slice(start:number, end:number) {
    // Returns a slice of the DataTable
    const values = this.values.map((v) => v.slice(start, end));
    return new DataTable(values, this.columns, this.index);
  }

  filter(condition: (v: any) => boolean) {
    // Returns a filtered version of the DataTable
    const values = this.values.map((v) => v.filter(condition));
    return new DataTable(values, this.columns, this.index);
  }

  sort(column: string, ascending: boolean = true) {
    // Returns a sorted version of the DataTable
    const values = this.values.map((v) => v.sort(column, ascending));
    return new DataTable(values, this.columns, this.index);
  }

  transpose() {
    // Returns a transposed version of the DataTable
    const index = this.columns;
    const columns = this.index;
    const curValues = this.values;
    const newShape: number[] = [this.shape[1], this.shape[0]];
    const newValues: any[] = [];
    for (let i = 0; i < newShape[0]; i++) {
      newValues.push([]);
    }
    curValues.forEach((v) => {
      for (let i = 0; i < v.length; i++) {
        newValues[i].push(v[i]);
      }
    });
    const newVals2 = newValues.map((v) => new DataSeries(v));

    return new DataTable(newVals2, columns, index);
  }

  mapColumns(func) {
    // Returns a mapped version of the DataTable
    const values = this.values.map((v) => v.map(func));
    return new DataTable(values, this.columns, this.index);
  }

  split(nChunks = 2, output = 'json', saveFile = '') {
    let dt = this.df();
    const nRows = dt.height;

    // add a row number column
    dt = dt.withColumn(pl.lit(1).cumSum().alias('rowNumber'));

    // add a column to indicate which chunk the row belongs to and one
    // to indicate the total number of chunks (so no ambiguity when
    // waiting for the remaining chunks)
    dt = dt
      .withColumn(
        pl.col('rowNumber').modulo(pl.lit(nChunks)).alias('chunkNumber'),
      )
      .withColumn(pl.lit(nChunks).alias('nChunks'));

    // split the dataframe into chunks
    const chunks = [];
    for (let i = 0; i < nChunks; i++) {
      const chunk = dt.filter(pl.col('chunkNumber').eq(pl.lit(i)));

      // save the chunks to files if requested
      if (saveFile !== '') {
        for (let i = 0; i < nChunks; i++) {
          const fileExtension = saveFile.split('.').pop();
          const chunkPath = `${saveFile}.${i}.${fileExtension}`;
          if (fileExtension === 'csv') {
            chunk.writeCSV(chunkPath);
          } else if (fileExtension === 'parquet') {
            chunk.writeParquet(chunkPath);
          } else if (fileExtension === 'json') {
            chunk.writeJSON(chunkPath);
          } else {
            throw new Error(`Unsupported file type: ${fileExtension}`);
          }
        }
      }

      // output the chunk in the requested format
      let chunkData;
      if (output === 'df') {
        chunkData = chunk;
      } else if (output === 'json') {
        chunkData = chunk.toJSON();
      } else {
        throw new Error(`Unknown output format: ${output}`);
      }
      chunks.push(chunkData);
    }

    return chunks;
  }

  static fromJSON(json) {
    // Returns a DataTable representation of the JSON object
    return new DataTable(json.id, json.name, json.columnName, json.data);
  }
}

export default DataTable;
