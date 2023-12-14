// Note: Table Model used to simplify the data structure, as well as have a way to convert back and forth between Polars and JSON,
// as well as a method to split the data table into smaller chunks for pagination.

import pl from 'nodejs-polars';

interface DataTableType {
  id: number;
  name: string;
  columnName: string[];
  data: string | Buffer;
}

class DataTable implements DataTableType {
  id: number;
  name: string;
  columnName: string[];
  data: string | Buffer;

  constructor(
    id: number,
    name: string,
    columnName: string[],
    data: string | Buffer,
  ) {
    this.id = id;
    this.name = name || '';
    this.columnName = columnName || [];
    this.data = data || '';
  }

  df(this: DataTable) {
    // Returns a Polars DataFrame representation of the DataTable
    return pl.readJSON(this.data.toString());
  }

  col(this: DataTable, column: string) {
    // Returns a single column by name as a polars Series
    const df = this.df();
    return df.getColumn(column);
  }

  cols(this: DataTable) {
    // Returns the DataTable as an array of polars series
    const df = this.df();
    return df.getColumns();
  }

  json(this: DataTable) {
    // Returns a JSON representation of the DataTable
    return this.data;
  }

  split(
    this: DataTable,
    nChunks: number,
    output: 'json' | 'df' = 'json',
    saveFile: string = '',
  ) {
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

  static fromPolars(id: number, name: string, df: any) {
    // Returns a DataTable representation of the Polars DataFrame
    return new DataTable(id, name, df.columns, df.toJSON());
  }
  static fromJSON(json: any) {
    // Returns a DataTable representation of the JSON object
    return new DataTable(json.id, json.name, json.columnName, json.data);
  }
  static async fromJSONStream(chunks: any, id: number, name: string) {
    // Returns a DataTable representation of the stream of JSON chunks
    const data: pl.DataFrame[] = [];
    let chunkCount = 0;

    // Will wait for each chunk to be received before continuing
    // ie will wait for the the total number of chunks to be
    // equal to the `nChunks` column from the first chunk
    for await (const chunk of chunks) {
      // if this is the first chunk, get the total number of chunks
      if (chunkCount === 0) {
        // create a data table from the first chunk
        const firstChunk = DataTable.fromJSON(chunk).df();
        const nChunks = firstChunk.select(pl.col('nChunks').first());
      }
      data.push(DataTable.fromJSON(chunk).df()); // add the chunk to the data
      chunkCount++; // increment the chunk count
      if (chunkCount === chunks.length) {
        break; // break if all chunks have been received
      }
    }

    // concatenate the chunks into a single dataframe
    const df = pl.concat(data);
    return new DataTable(id, name, df.columns, df.toJSON());
  }
}

export default DataTable;
