import DataTable from '@models/DataTable/DataTable';

interface TableProps {
  dt: DataTable | null;
  maxRows?: number;
  maxCols?: number;
  renderIndex?: boolean;
}

/**
 * @component Table
 * @summary Renders a table of data from a DataTable object based on the original .render() method of a DataTable copied above.
 */
const Table = ({
  dt,
  maxRows = -1,
  maxCols = -1,
  renderIndex = true,
}: TableProps) => {
  // Returns an HTML table representation of the DataTable

  if (!dt) {
    return <div>dt is null</div>;
  } else {
    const dt1 = dt.transpose();
    const { index, columns, values } = dt1;

    const nRows = maxRows === -1 ? index.length : maxRows;
    const nCols = maxCols === -1 ? columns.length : maxCols;

    // Get the first maxRows rows of the index
    const renderedIndex = index.slice(0, nRows);

    // Get the first maxCols columns of the values
    const renderedCols = columns.slice(0, nCols);

    // Get the first maxRows rows x maxCols columns of the values
    const renderedValues = values.map((v: any) => v.slice(0, nRows, nCols));
    console.log('renderedValues:', renderedValues);

    // Get the maximum length of the index and column names
    const maxIndexLength = Math.max(...renderedIndex.map((i: any) => i.length));
    const maxColLength = Math.max(...renderedCols.map((c: any) => c.length));

    // Pad the index and column names with spaces to make them all the same length
    renderedIndex.forEach((i: any, j: number) => {
      renderedIndex[j] = String(i).padEnd(maxIndexLength, ' ');
    });
    renderedCols.forEach((c: any, j: number) => {
      renderedCols[j] = String(c).padEnd(maxColLength, ' ');
    });

    return (
      <table className="h-full w-full text-sm">
        <thead>
          <tr>
            {renderIndex && <th> </th>}{' '}
            {/* If you want to render the index as a header */}
            {renderedCols.map((col: any, index: number) => (
              <th
                key={index}
                className="text-right border-black border-[2px] font-semibold bg-slate-400"
              >
                {String(col).replace('_', ' ')}
              </th> // Make sure to return the <th> element
            ))}
          </tr>
        </thead>
        <tbody>
          {
            // Mapping
            renderedValues.map((row: any, rowIndex: number) => {
              return (
                <tr key={rowIndex}>
                  {renderIndex && <td>{index[rowIndex]}</td>}{' '}
                  {/* Render index cell */}
                  {row.map((val: any, colIndex: number) => (
                    <td key={colIndex}>{val}</td> // Render value cells
                  ))}
                </tr>
              );
            })
          }
        </tbody>
      </table>
    );
  }
};

export default Table;
