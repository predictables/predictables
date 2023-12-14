// DataTable class is currently an array of DataSeries objects, with some additional methods
// Particular interest in the .render() method, which returns a formatted HTML table
import DataSeries, { dtypeTypes } from './DataSeries';

class DataTable {
  constructor(values, columns = [], index = null) {
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
      values.forEach((v) => (v.index = index));
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
  }

  col(column) {
    // Returns a single column by name as a DataSeries
    const index = this.columns.indexOf(column);
    if (index === -1) {
      throw new Error(`Column ${column} not found`);
    }
    return this.values[index];
  }

  json(byRow = false) {
    // Returns a JSON representation of the DataTable
    const json = {};
    json.index = this.index;
    this.columns.forEach((c, i) => {
      json[c] = this.values[i].json(byRow);
    });
    return json;
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
