// // DataTable class is currently an array of DataSeries objects, with some additional methods
// // Particular interest in the .render() method, which returns a formatted HTML table
// import DataSeries, { dtypeTypes } from '../DataSeries';
// import { extract } from '@utils/extract';

// interface DataTableProps {
//   values: DataSeries[]; // array of DataSeries, each one representing a column
//   columns?: string[]; // column names - defaut to empty array, which will be filled in from the data
//   index?: any[] | null; // index values - default to null, which will be filled in from the data
// }

// type dtypeTypes = (typeof dtypeTypes)[number];

// class DataTable {
//   values: DataSeries[];
//   columns: string[];
//   index: any[] | null;
//   dtypes: dtypeTypes[];
//   shape: [number, number];

//   constructor(
//     values: DataSeries[],
//     columns: string[] = [],
//     index: any[] | null = null,
//   ) {
//     // ensure each element of the values array is a DataSeries
//     if (values.length === 0) {
//       throw new Error('Values array must not be empty');
//     }

//     // ensure all values are DataSeries
//     if (values.some((v) => !(v instanceof DataSeries))) {
//       throw new Error('Values array must contain only DataSeries objects');
//     }

//     // ensure all values are the same length
//     const lengths = values.map((v) => v.length);
//     if (lengths.some((l) => l !== lengths[0])) {
//       throw new Error('All DataSeries must be the same length');
//     }

//     // if an index was not passed, ensure all values have the same index array, and if not, reset all of them to the default
//     if (index === null) {
//       const indices = values.map((v) => v.index);

//       // test if all indices are elementwise equal
//       const cond = (i: any, j: any) => i === j;
//       const indicesEqual = indices.every((i) =>
//         indices.every((j) => cond(i, j)),
//       );

//       // if they are not all equal, reset them all to the default, which is the index of the first value
//       if (!indicesEqual) {
//         const defaultIndex = values[0].index;
//         values.forEach((v) => (v.index = defaultIndex));
//       }

//       // set the index to the index of the first value
//       this.index = values[0].index;
//     } else {
//       // set every index to be the passed index, as well as the class index
//       values.forEach((v) => (v.index = index));
//       this.index = index;
//     }

//     // if a columns array was not passed, set it to the names of the values
//     if (columns.length === 0) {
//       values.forEach((v, i) => {
//         // ensure all values have a name -- if not give them a default name (column0, column1, etc)
//         if (v.name === null || v.name === '') {
//           v.name = `column${i}`;
//         }
//       });

//       // ensure all the column names are unique
//       let colNames: any[] = values.map((v) => v.name);
//       let cols = new DataSeries(colNames);
//       if (cols.shape[0] !== cols.unique().shape[0]) {
//         throw new Error('Column names must be unique');
//       }

//       // set the columns to the names of the values
//       this.columns = colNames;
//     } else {
//       // ensure the columns array is the same length as the values array
//       if (columns.length !== values.length) {
//         throw new Error(
//           'Columns array must be the same length as the values array',
//         );
//       }
//     }
//   }

//   // df(this: DataTable) {
//   //   // Returns a Polars DataFrame representation of the DataTable
//   //   return pl.readJSON(this.data.toString());
//   // }

//   col(this: DataTable, column: string) {
//     // Returns a single column by name as a polars Series
//     const df = this.df();
//     return df.getColumn(column);
//   }

//   cols(this: DataTable) {
//     // Returns the DataTable as an array of polars series
//     const df = this.df();
//     return df.getColumns();
//   }

//   json(this: DataTable) {
//     // Returns a JSON representation of the DataTable
//     return this.data;
//   }

//   split(
//     this: DataTable,
//     nChunks: number,
//     output: 'json' | 'df' = 'json',
//     saveFile: string = '',
//   ) {
//     let dt = this.df();
//     const nRows = dt.height;

//     // add a row number column
//     dt = dt.withColumn(pl.lit(1).cumSum().alias('rowNumber'));

//     // add a column to indicate which chunk the row belongs to and one
//     // to indicate the total number of chunks (so no ambiguity when
//     // waiting for the remaining chunks)
//     dt = dt
//       .withColumn(
//         pl.col('rowNumber').modulo(pl.lit(nChunks)).alias('chunkNumber'),
//       )
//       .withColumn(pl.lit(nChunks).alias('nChunks'));

//     // split the dataframe into chunks
//     const chunks = [];
//     for (let i = 0; i < nChunks; i++) {
//       const chunk = dt.filter(pl.col('chunkNumber').eq(pl.lit(i)));

//       // save the chunks to files if requested
//       if (saveFile !== '') {
//         for (let i = 0; i < nChunks; i++) {
//           const fileExtension = saveFile.split('.').pop();
//           const chunkPath = `${saveFile}.${i}.${fileExtension}`;
//           if (fileExtension === 'csv') {
//             chunk.writeCSV(chunkPath);
//           } else if (fileExtension === 'parquet') {
//             chunk.writeParquet(chunkPath);
//           } else if (fileExtension === 'json') {
//             chunk.writeJSON(chunkPath);
//           } else {
//             throw new Error(`Unsupported file type: ${fileExtension}`);
//           }
//         }
//       }

//       // output the chunk in the requested format
//       let chunkData;
//       if (output === 'df') {
//         chunkData = chunk;
//       } else if (output === 'json') {
//         chunkData = chunk.toJSON();
//       } else {
//         throw new Error(`Unknown output format: ${output}`);
//       }
//       chunks.push(chunkData);
//     }

//     return chunks;
//   }

//   static fromPolars(id: number, name: string, df: any) {
//     // Returns a DataTable representation of the Polars DataFrame
//     return new DataTable(id, name, df.columns, df.toJSON());
//   }
//   static fromJSON(json: any) {
//     // Returns a DataTable representation of the JSON object
//     return new DataTable(json.id, json.name, json.columnName, json.data);
//   }
//   static async fromJSONStream(chunks: any, id: number, name: string) {
//     // Returns a DataTable representation of the stream of JSON chunks
//     const data: pl.DataFrame[] = [];
//     let chunkCount = 0;

//     // Will wait for each chunk to be received before continuing
//     // ie will wait for the the total number of chunks to be
//     // equal to the `nChunks` column from the first chunk
//     for await (const chunk of chunks) {
//       // if this is the first chunk, get the total number of chunks
//       if (chunkCount === 0) {
//         // create a data table from the first chunk
//         const firstChunk = DataTable.fromJSON(chunk).df();
//         const nChunks = firstChunk.select(pl.col('nChunks').first());
//       }
//       data.push(DataTable.fromJSON(chunk).df()); // add the chunk to the data
//       chunkCount++; // increment the chunk count
//       if (chunkCount === chunks.length) {
//         break; // break if all chunks have been received
//       }
//     }

//     // concatenate the chunks into a single dataframe
//     const df = pl.concat(data);
//     return new DataTable(id, name, df.columns, df.toJSON());
//   }
// }

// export default DataTable;
